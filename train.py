import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from flax import linen as nn
import optax
from flax.training import train_state, checkpoints
from typing import Any
import tqdm

class ControlDataset:
    def __init__(self, file_path, sequence_length=256, scale_data=True):
        df = pd.read_csv(file_path)
        self.condition_columns = ['pos_d[0]', 'pos_d[1]', 'pos_d[2]', 'yaw_d']
        self.target_columns = ['ang_vel[0]', 'ang_vel[1]', 'ang_vel[2]',
                             'acc[0]', 'acc[1]', 'acc[2]',
                             'omega[0]', 'omega[1]', 'omega[2]', 'omega[3]']
        
        self.condition_data = df[self.condition_columns].values
        self.target_data = df[self.target_columns].values
        self.sequence_length = sequence_length
        self.feature_dim = len(self.target_columns)
        self.condition_dim = len(self.condition_columns)
        
        if scale_data:
            self.scaler = StandardScaler()
            self.target_data = self.scaler.fit_transform(self.target_data)

    def __len__(self):
        return (len(self.target_data) - (self.sequence_length + 1)) // self.sequence_length

    def get_batch(self, indices):
        inputs, targets, conditions = [], [], []
        for idx in indices:
            start_idx = idx * self.sequence_length
            inputs.append(self.target_data[start_idx:start_idx + self.sequence_length])
            targets.append(self.target_data[start_idx + 1:start_idx + self.sequence_length + 1])
            conditions.append(self.condition_data[start_idx:start_idx + self.sequence_length])
        return (jnp.array(inputs), jnp.array(targets), jnp.array(conditions))

class AdaptiveRMSNorm(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x, condition):
        scale = nn.Dense(1)(nn.Dense(self.hidden_dim)(condition))
        scale = jax.nn.sigmoid(scale) * 2
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(variance + 1e-5) * self.param('weight', nn.initializers.ones, (self.hidden_dim,)) * scale

class TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, condition, training=True):
        attn_out = x + nn.MultiHeadDotProductAttention(num_heads=self.num_heads, dropout_rate=self.dropout_rate)(
            inputs_q=AdaptiveRMSNorm(self.hidden_dim)(x, condition),
            inputs_kv=x,
            deterministic=not training)
        
        dense_out = nn.Dense(self.hidden_dim)(
            nn.Dropout(rate=self.dropout_rate)(
                nn.gelu(nn.Dense(self.hidden_dim * 4)(
                    AdaptiveRMSNorm(self.hidden_dim)(attn_out, condition))),
                deterministic=not training))
        
        return attn_out + dense_out

class ControlTransformer(nn.Module):
    num_layers: int
    hidden_dim: int
    num_heads: int
    dropout_rate: float
    feature_dim: int

    @nn.compact
    def __call__(self, x, condition, training=True):
        x = nn.Dense(self.hidden_dim)(x) + self.param('pos_embedding', 
            nn.initializers.normal(0.02), (256, self.hidden_dim))[:x.shape[1]]
        
        for _ in range(self.num_layers):
            x = TransformerBlock(self.hidden_dim, self.num_heads, self.dropout_rate)(x, condition, training)
        
        return nn.Dense(self.feature_dim)(AdaptiveRMSNorm(self.hidden_dim)(x, condition))

class TrainState(train_state.TrainState):
    dropout_rng: Any

@jax.jit
def train_step(state, batch):
    inputs, targets, conditions = batch
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

    def loss_fn(params):
        predictions = state.apply_fn({'params': params}, inputs, conditions, 
                                   training=True, rngs={'dropout': dropout_rng})
        return jnp.mean((predictions - targets) ** 2), predictions

    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads).replace(dropout_rng=new_dropout_rng), {'loss': loss}

@jax.jit
def eval_step(state, batch):
    predictions = state.apply_fn({'params': state.params}, batch[0], batch[2], training=False)
    return {'loss': jnp.mean((predictions - batch[1]) ** 2)}

def train(config):
    rng = jax.random.PRNGKey(config['seed'])
    dataset = ControlDataset(config['data_path'], config['seq_len'])
    train_size = int(len(dataset) * (1 - config['val_split']))
    
    model = ControlTransformer(
        num_layers=config['num_layers'], hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'], dropout_rate=config['dropout_rate'],
        feature_dim=dataset.feature_dim)
    
    variables = model.init(rng, 
        jnp.ones((1, config['seq_len'], dataset.feature_dim)),
        jnp.ones((1, config['seq_len'], dataset.condition_dim)), training=True)
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(config['learning_rate']),
        dropout_rng=rng)

    for epoch in range(config['num_epochs']):
        # Training
        indices = np.arange(train_size)
        np.random.shuffle(indices)
        
        train_losses = []
        progress_bar = tqdm.tqdm(
            range(0, train_size, config['batch_size']),
            desc=f"Epoch {epoch+1}/{config['num_epochs']}",
            total=(train_size // config['batch_size'])
        )
        
        for i in progress_bar:
            batch_indices = indices[i:i + config['batch_size']]
            if len(batch_indices) < config['batch_size']:
                continue
            batch = dataset.get_batch(batch_indices)
            state, metrics = train_step(state, batch)
            train_losses.append(metrics['loss'])
            
            # Update progress bar with current loss
            avg_loss = np.mean(train_losses[-50:] if len(train_losses) > 50 else train_losses)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
        # Validation
        val_losses = []
        val_indices = np.arange(train_size, len(dataset))
        val_progress = tqdm.tqdm(
            range(0, len(val_indices), config['batch_size']),
            desc="Validation",
            total=(len(val_indices) // config['batch_size'])
        )
        
        for i in val_progress:
            batch_indices = val_indices[i:i + config['batch_size']]
            if len(batch_indices) < config['batch_size']:
                continue
            batch = dataset.get_batch(batch_indices)
            metrics = eval_step(state, batch)
            val_losses.append(metrics['loss'])
            val_progress.set_postfix({'val_loss': f'{np.mean(val_losses):.4f}'})
            
        print(f"Epoch {epoch + 1} - Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}")
        
        if (epoch + 1) % config['save_every'] == 0:
            checkpoints.save_checkpoint("checkpoints", state, epoch + 1, keep=3)

if __name__ == "__main__":
    config = {
        'data_path': '2024-12-29_6-25-1_control_data.csv',
        'val_split': 0.1,
        'seq_len': 256,
        'batch_size': 32,
        'num_layers': 6,
        'num_heads': 8,
        'hidden_dim': 384,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'dropout_rate': 0.1,
        'save_every': 5,
        'seed': 42
    }
    train(config)