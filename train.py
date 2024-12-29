import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
import tqdm
import random
import optax

# ============= PART 1: CORE MODEL COMPONENTS =============

class ControlDataset:
    def __init__(self, file_path, sequence_length=256, scale_data=True):
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        self.target_columns = ['ang_vel[0]', 'ang_vel[1]', 'ang_vel[2]',
                             'acc[0]', 'acc[1]', 'acc[2]',
                             'omega[0]', 'omega[1]', 'omega[2]', 'omega[3]']
        self.data = df[self.target_columns].values
        self.sequence_length = sequence_length
        self.feature_dim = len(self.target_columns)
        
        if scale_data:
            scaler_path = f"{file_path}.scaler"
            self.scaler = (pickle.load(open(scaler_path, 'rb')) if os.path.exists(scaler_path) 
                          else self._fit_save_scaler(scaler_path))
            self.data = self.scaler.transform(self.data)

    def _fit_save_scaler(self, path):
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)
        with open(path, 'wb') as f:
            pickle.dump(scaler, f)
        return scaler

    def __len__(self):
        return (len(self.data) - (self.sequence_length + 1)) // self.sequence_length

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        inputs = jnp.array(self.data[start_idx:start_idx + self.sequence_length], dtype=jnp.float32)
        labels = jnp.array(self.data[start_idx + 1:start_idx + self.sequence_length + 1], dtype=jnp.float32)
        return inputs, labels

def create_train_val_datasets(file_path, sequence_length=256, val_split=0.1):
    dataset = ControlDataset(file_path, sequence_length)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    indices = list(range(len(dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = ControlDataset(file_path, sequence_length, False)
    val_dataset = ControlDataset(file_path, sequence_length, False)
    
    train_dataset.data = dataset.data[:train_size * sequence_length]
    val_dataset.data = dataset.data[train_size * sequence_length:]
    
    return train_dataset, val_dataset

def get_batches(dataset, batch_size):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        if len(batch_indices) < batch_size:
            continue
            
        inputs = []
        targets = []
        for idx in batch_indices:
            x, y = dataset[idx]
            inputs.append(x)
            targets.append(y)
            
        yield jnp.stack(inputs), jnp.stack(targets)

def attention(params, x, mask, batch_size, seq_len, num_heads, hidden_dim):
    head_dim = hidden_dim // num_heads
    def process_qkv(input_data, weight):
        return (jnp.dot(input_data, weight)
                .reshape((batch_size, seq_len, num_heads, head_dim))
                .transpose(0, 2, 1, 3))
    
    q = process_qkv(x, params['q_linear'])
    k = process_qkv(x, params['k_linear'])
    v = process_qkv(x, params['v_linear'])
    
    scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * (head_dim ** -0.5)
    scores = jax.nn.softmax(scores + mask, axis=3)
    
    output = (jnp.matmul(scores, v)
             .transpose(0, 2, 1, 3)
             .reshape((batch_size, seq_len, hidden_dim)))
    return jnp.dot(output, params['o_linear'])

def simple_rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + 1e-5)

def feed_forward(params, x):
    x = jnp.dot(x, params['in_weight'])
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.dot(x1 * jax.nn.sigmoid(x2), params['out_weight'])

def transformer_block(params, x, mask, batch_size, seq_len, num_heads, hidden_dim):
    x = x + attention(params['attention'], simple_rms_norm(x), mask, 
                     batch_size, seq_len, num_heads, hidden_dim)
    x = x + feed_forward(params['feed_forward'], simple_rms_norm(x))
    return x

def forward(params, x):
    batch_size, seq_len = x.shape[:2]
    x = jnp.dot(x, params['input_projection'])
    x = x + params['positional_embedding'][:seq_len]
    
    mask = jnp.where(jnp.tril(jnp.ones((seq_len, seq_len))) == 0, -1e10, 0)[None, None, :, :]
    
    for block in params['transformer_blocks']:
        x = transformer_block(block, x, mask, batch_size, seq_len, 
                            num_heads=8, hidden_dim=384)
    
    return jnp.dot(simple_rms_norm(x), params['output_projection'])

def init_params(feature_dim, seq_len, num_blocks=6, num_heads=8, hidden_dim=384, ff_dim=1536, dtype=jnp.float32):
    rng_key = jax.random.PRNGKey(0)
    keys = jax.random.split(rng_key, num_blocks * 6 + 4)
    key_idx = 0
    
    def next_key():
        nonlocal key_idx
        key = keys[key_idx]
        key_idx += 1
        return key
    
    xavier_init = jax.nn.initializers.glorot_uniform(dtype=dtype)
    kaiming_init = jax.nn.initializers.he_normal(dtype=dtype)
    
    params = {
        'input_projection': xavier_init(next_key(), (feature_dim, hidden_dim)),
        'output_projection': xavier_init(next_key(), (hidden_dim, feature_dim)),
        'positional_embedding': jax.random.normal(next_key(), (256, hidden_dim)) * 0.02,
        'transformer_blocks': [{
            'attention': {
                'q_linear': xavier_init(next_key(), (hidden_dim, hidden_dim)),
                'k_linear': xavier_init(next_key(), (hidden_dim, hidden_dim)),
                'v_linear': xavier_init(next_key(), (hidden_dim, hidden_dim)),
                'o_linear': xavier_init(next_key(), (hidden_dim, hidden_dim)),
            },
            'feed_forward': {
                'in_weight': kaiming_init(next_key(), (hidden_dim, ff_dim)),
                'out_weight': xavier_init(next_key(), (ff_dim // 2, hidden_dim)),
            }
        } for _ in range(num_blocks)]
    }
    
    return params

# ============= PART 2: TRAINING INFRASTRUCTURE =============

@jax.jit
def train_step(params, opt_state, batch):
    inputs, targets = batch
    
    def loss_fn(params):
        pred = forward(params, inputs)
        loss = jnp.mean((pred - targets) ** 2)
        return loss, pred
    
    (loss, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    metrics = {
        'loss': loss,
        'mse': jnp.mean((pred - targets) ** 2, axis=(0, 1)),
        'mae': jnp.mean(jnp.abs(pred - targets), axis=(0, 1))
    }
    
    return grads, metrics

def evaluate(params, dataset, batch_size):
    metrics_list = []
    for batch in get_batches(dataset, batch_size):
        inputs, targets = batch
        pred = forward(params, inputs)
        metrics = {
            'loss': jnp.mean((pred - targets) ** 2),
            'mse': jnp.mean((pred - targets) ** 2, axis=(0, 1)),
            'mae': jnp.mean(jnp.abs(pred - targets), axis=(0, 1))
        }
        metrics_list.append(metrics)
    
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}

def save_checkpoint(params, filename):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved checkpoint to {filename}")

def train(config):
    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        config['data_path'], 
        config['seq_len'], 
        config['val_split']
    )
    
    # Initialize model and optimizer
    params = init_params(
        feature_dim=train_dataset.feature_dim,
        seq_len=config['seq_len'],
        num_blocks=config['num_blocks'],
        num_heads=config['num_heads'],
        hidden_dim=config['hidden_dim'],
        ff_dim=config['ff_dim']
    )
    
    optimizer = optax.adam(config['learning_rate'])
    opt_state = optimizer.init(params)
    
    # Calculate total number of batches
    total_batches = len(train_dataset) // config['batch_size']
    
    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        train_metrics = []
        
        # Create progress bar for batches
        progress_bar = tqdm.tqdm(
            get_batches(train_dataset, config['batch_size']),
            total=total_batches,
            desc=f"Training",
            leave=True,
            ncols=100
        )
        
        for batch in progress_bar:
            grads, metrics = train_step(params, opt_state, batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            train_metrics.append(metrics)
            
            # Update progress bar with current loss
            current_loss = np.mean([m['loss'] for m in train_metrics[-10:]])  # Moving average of last 10 batches
            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}"
            })
        
        # Validation
        val_metrics = evaluate(params, val_dataset, config['batch_size'])
        
        # Print epoch summary
        mean_train_loss = np.mean([m['loss'] for m in train_metrics])
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {mean_train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        
        # Save checkpoint
        if epoch % config['save_every'] == 0:
            save_checkpoint(params, f"checkpoint_epoch_{epoch}.pkl")

if __name__ == "__main__":
    config = {
        'data_path': '2024-12-29_6-25-1_control_data.csv',
        'val_split': 0.1,
        'seq_len': 256,
        'batch_size': 32,
        'num_blocks': 6,
        'num_heads': 8,
        'hidden_dim': 384,
        'ff_dim': 1536,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'save_every': 5,
        'seed': 42
    }
    
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    train(config)