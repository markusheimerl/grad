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

class ControlDataset:
    def __init__(self, file_path, sequence_length=256, scale_data=True):
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        self.target_columns = [
            'ang_vel[0]', 'ang_vel[1]', 'ang_vel[2]',
            'acc[0]', 'acc[1]', 'acc[2]',
            'omega[0]', 'omega[1]', 'omega[2]', 'omega[3]'
        ]
        
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

def create_train_val_datasets(file_path, sequence_length=256, val_split=0.1, scale_data=True):
    dataset = ControlDataset(file_path, sequence_length, scale_data)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_data = dataset.data[:train_size * sequence_length]
    val_data = dataset.data[train_size * sequence_length:]
    
    train_dataset = ControlDataset(file_path, sequence_length, False)
    val_dataset = ControlDataset(file_path, sequence_length, False)
    train_dataset.data, val_dataset.data = train_data, val_data
    train_dataset.scaler = val_dataset.scaler = dataset.scaler
    
    return train_dataset, val_dataset

def attention(params, x, mask, batch_size, seq_len, num_heads, hidden_dim):
    head_dim = hidden_dim // num_heads
    
    def process_qkv(input_data, weight):
        return (jax.numpy.dot(input_data, weight)
                .reshape((batch_size, seq_len, num_heads, head_dim))
                .transpose(0, 2, 1, 3))
    
    q = process_qkv(x, params['q_linear'])
    k = process_qkv(x, params['k_linear'])
    v = process_qkv(x, params['v_linear'])
    
    scores = jax.numpy.matmul(q, k.transpose(0, 1, 3, 2)) * (head_dim ** -0.5)
    scores = jax.nn.softmax(scores + mask, axis=3)
    
    output = (jax.numpy.matmul(scores, v)
             .transpose(0, 2, 1, 3)
             .reshape((batch_size, seq_len, hidden_dim)))
    return jax.numpy.dot(output, params['o_linear'])

def transformer_block(params, x, mask, batch_size, seq_len, num_heads, hidden_dim):
    x = x + attention(params['attention'], simple_rms_norm(x), mask, 
                     batch_size, seq_len, num_heads, hidden_dim)
    x = x + feed_forward(params['feed_forward'], simple_rms_norm(x))
    return x

def simple_rms_norm(x):
    return x * jax.lax.rsqrt(jax.numpy.mean(jax.numpy.square(x), axis=2, keepdims=True) + 1e-5)

def feed_forward(params, x):
    x = jax.numpy.dot(x, params['in_weight'])
    x1, x2 = jax.numpy.split(x, 2, axis=2)
    return jax.numpy.dot(x1 * jax.nn.sigmoid(x2), params['out_weight'])

def init_params(feature_dim, seq_len, num_blocks=6, num_heads=8, hidden_dim=384, ff_dim=1536, dtype=jnp.float32, rng_key=jax.random.key(0)):
    keys = jax.random.split(rng_key, num_blocks * 6 + 4)
    key_idx = 0
    
    def get_next_key():
        nonlocal key_idx
        key = keys[key_idx]
        key_idx += 1
        return key
    
    xavier_init = jax.nn.initializers.glorot_uniform(dtype=dtype)
    kaiming_init = jax.nn.initializers.he_normal(dtype=dtype)
    
    learnable_params = {
        'input_projection': xavier_init(get_next_key(), (feature_dim, hidden_dim)),
        'output_projection': xavier_init(get_next_key(), (hidden_dim, feature_dim)),
        'positional_embedding': jax.random.normal(get_next_key(), (256, hidden_dim)) * 0.02,
        'transformer_blocks': [{
            'attention': {
                'q_linear': xavier_init(get_next_key(), (hidden_dim, hidden_dim)),
                'k_linear': xavier_init(get_next_key(), (hidden_dim, hidden_dim)),
                'v_linear': xavier_init(get_next_key(), (hidden_dim, hidden_dim)),
                'o_linear': xavier_init(get_next_key(), (hidden_dim, hidden_dim)),
            },
            'feed_forward': {
                'in_weight': kaiming_init(get_next_key(), (hidden_dim, ff_dim)),
                'out_weight': xavier_init(get_next_key(), (ff_dim // 2, hidden_dim)),
            }
        } for _ in range(num_blocks)]
    }
    
    mask = jnp.where(jnp.tril(jnp.ones((256, 256))) == 0, -1e10, 0)[None, None, :, :]
    
    return learnable_params, {"mask": mask, "num_heads": num_heads, "hidden_dim": hidden_dim}

def create_model(feature_dim, seq_len=256, batch_size=32, num_blocks=8, num_heads=8, hidden_dim=512, ff_dim=2048, dtype=jnp.float32, seed=0):
    learnable_params, static_config = init_params(
        feature_dim, seq_len, num_blocks, num_heads, hidden_dim, ff_dim, dtype, jax.random.key(seed)
    )
    return {
        'learnable_params': learnable_params,
        'static_config': static_config,
        'model_config': locals()
    }

@jax.jit
def train_step(learnable_params, optimizer_state, inputs, targets, mask):
    def loss_fn(params):
        predictions = control_model(params, inputs, mask, 32, 256, 8, 384)
        return jax.numpy.mean(jax.numpy.square(predictions - targets)), predictions
    
    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(learnable_params)
    learnable_params, optimizer_state = apply_adam_optimizer(learnable_params, optimizer_state, grads)
    return learnable_params, optimizer_state, loss, compute_metrics(predictions, targets)

def train(config):
    train_dataset, val_dataset = create_train_val_datasets(
        config['data_path'], config['seq_len'], config['val_split'], True
    )
    
    model = create_model(
        train_dataset.feature_dim,
        **{k: config[k] for k in ['seq_len', 'batch_size', 'num_blocks', 
                                 'num_heads', 'hidden_dim', 'ff_dim', 'seed']}
    )
    
    optimizer_state = create_adam_state(model['learnable_params'], config['learning_rate'])
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Training
        indices = random.sample(range(len(train_dataset)), len(train_dataset))
        train_metrics = {'losses': [], 'mse': [], 'mae': []}
        
        for i in tqdm.tqdm(range(0, len(indices), config['batch_size'])):
            inputs, targets = prepare_batch_data(train_dataset, 
                                              indices[i:i + config['batch_size']], 
                                              config['batch_size'])
            if inputs is None:
                continue
                
            model['learnable_params'], optimizer_state, loss, metrics = train_step(
                model['learnable_params'], optimizer_state, inputs, targets, 
                model['static_config']['mask']
            )
            
            train_metrics['losses'].append(loss)
            train_metrics['mse'].append(metrics['mse'])
            train_metrics['mae'].append(metrics['mae'])
        
        # Validation
        val_metrics = evaluate_model(model, val_dataset, config['batch_size'])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(model, f"checkpoint_{epoch}_{val_metrics['loss']:.6f}.npz")
        
        print_metrics("Training", train_metrics)
        print_metrics("Validation", val_metrics)

def print_metrics(phase, metrics):
    print(f"\n{phase} Metrics:")
    print(f"Loss: {np.mean(metrics['losses']):.6f}")
    print(f"MSE: {np.mean([m.mean() for m in metrics['mse']]):.6f}")
    print(f"MAE: {np.mean([m.mean() for m in metrics['mae']]):.6f}")

def create_adam_state(learnable_params, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    """Initialize Adam optimizer state"""
    return {
        "step": 0,
        "learning_rate": learning_rate,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "epsilon": epsilon,
        "m": jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), learnable_params),
        "v": jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), learnable_params)
    }

def apply_adam_optimizer(learnable_params, adam_state, grads):
    """Apply Adam optimization step"""
    adam_state['step'] += 1
    
    # Update momentum and velocity
    adam_state['m'] = jax.tree_util.tree_map(
        lambda m, g: adam_state['beta_1'] * m + (1 - adam_state['beta_1']) * g,
        adam_state['m'], grads
    )
    adam_state['v'] = jax.tree_util.tree_map(
        lambda v, g: adam_state['beta_2'] * v + (1 - adam_state['beta_2']) * (g ** 2),
        adam_state['v'], grads
    )
    
    # Compute bias-corrected moments
    m_hat = jax.tree_util.tree_map(
        lambda m: m / (1 - adam_state['beta_1'] ** adam_state['step']),
        adam_state['m']
    )
    v_hat = jax.tree_util.tree_map(
        lambda v: v / (1 - adam_state['beta_2'] ** adam_state['step']),
        adam_state['v']
    )
    
    # Compute updates
    updates = jax.tree_util.tree_map(
        lambda m, v: adam_state['learning_rate'] * m / (jnp.sqrt(v) + adam_state['epsilon']),
        m_hat, v_hat
    )
    
    # Apply updates
    learnable_params = jax.tree_util.tree_map(
        lambda p, u: p - u,
        learnable_params, updates
    )
    
    return learnable_params, adam_state

def control_model(params, inputs, mask, batch_size, seq_len, num_heads, hidden_dim):
    """Main model for control data prediction"""
    x = jnp.dot(inputs, params['input_projection'])
    x = x + params['positional_embedding']
    
    for block_params in params['transformer_blocks']:
        x = transformer_block(block_params, x, mask, batch_size, seq_len, num_heads, hidden_dim)
    
    return jnp.dot(simple_rms_norm(x), params['output_projection'])

def evaluate_model(model, dataset, batch_size):
    """Evaluate model on dataset"""
    metrics = {'losses': [], 'mse': [], 'mae': []}
    
    for i in range(0, len(dataset), batch_size):
        inputs, targets = prepare_batch_data(
            dataset,
            list(range(i, min(i + batch_size, len(dataset)))),
            batch_size
        )
        
        if inputs is None:
            continue
            
        loss, predictions = loss_fn(
            model['learnable_params'],
            inputs,
            targets,
            model['static_config']['mask']
        )
        
        batch_metrics = compute_metrics(predictions, targets)
        metrics['losses'].append(loss)
        metrics['mse'].append(batch_metrics['mse'])
        metrics['mae'].append(batch_metrics['mae'])
    
    return {
        'loss': np.mean(metrics['losses']),
        'mse': metrics['mse'],
        'mae': metrics['mae']
    }

def save_checkpoint(model, path):
    """Save model checkpoint"""
    jnp.savez(
        path,
        learnable_params=model['learnable_params'],
        static_config_mask=model['static_config']['mask'],
        static_config_num_heads=model['static_config']['num_heads'],
        static_config_hidden_dim=model['static_config']['hidden_dim'],
        model_config=model['model_config']
    )
    print(f"\nSaved checkpoint: {path}")

def prepare_batch_data(dataset, indices, batch_size):
    """
    Prepare batch data from dataset given indices
    
    Args:
        dataset: Dataset object
        indices: List of indices to use
        batch_size: Desired batch size
    
    Returns:
        tuple: (inputs, targets) or (None, None) if batch can't be created
    """
    if len(indices) < batch_size:
        return None, None
    
    inputs = []
    targets = []
    
    for idx in indices[:batch_size]:
        x, y = dataset[idx]
        inputs.append(x)
        targets.append(y)
    
    # Stack into batch
    inputs = jnp.stack(inputs)
    targets = jnp.stack(targets)
    
    return inputs, targets

def compute_metrics(predictions, targets):
    """
    Compute MSE and MAE metrics
    
    Args:
        predictions: Model predictions
        targets: True values
    
    Returns:
        dict: Dictionary containing MSE and MAE metrics
    """
    mse = jnp.mean(jnp.square(predictions - targets), axis=(0, 1))
    mae = jnp.mean(jnp.abs(predictions - targets), axis=(0, 1))
    return {'mse': mse, 'mae': mae}

def loss_fn(learnable_params, inputs, targets, mask):
    """
    Compute loss and predictions
    
    Args:
        learnable_params: Model parameters
        inputs: Input sequences
        targets: Target sequences
        mask: Attention mask
    
    Returns:
        tuple: (loss, predictions)
    """
    predictions = control_model(
        learnable_params, 
        inputs, 
        mask,
        batch_size=32,
        seq_len=256,
        num_heads=8,
        hidden_dim=384
    )
    loss = jnp.mean(jnp.square(predictions - targets))
    return loss, predictions

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
        'seed': random.randint(0, 2**16-1),
    }
    
    print("Training configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")
    
    random.seed(config['seed'])
    train(config)