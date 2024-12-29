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
from typing import Any
from flax import linen as nn
import optax
from flax.training import train_state
from flax.training import checkpoints

# ============= PART 1: DATASET =============

class ControlDataset:
    def __init__(self, file_path, sequence_length=256, scale_data=True):
        print(f"Loading data from {file_path}...")
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
            scaler_path = f"{file_path}.scaler"
            self.scaler = (pickle.load(open(scaler_path, 'rb')) if os.path.exists(scaler_path) 
                          else self._fit_save_scaler(scaler_path))
            self.target_data = self.scaler.transform(self.target_data)

    def _fit_save_scaler(self, path):
        scaler = StandardScaler()
        self.target_data = scaler.fit_transform(self.target_data)
        with open(path, 'wb') as f:
            pickle.dump(scaler, f)
        return scaler

    def __len__(self):
        return (len(self.target_data) - (self.sequence_length + 1)) // self.sequence_length

    def __getitem__(self, idx):
        start_idx = idx * self.sequence_length
        inputs = jnp.array(self.target_data[start_idx:start_idx + self.sequence_length], dtype=jnp.float32)
        labels = jnp.array(self.target_data[start_idx + 1:start_idx + self.sequence_length + 1], dtype=jnp.float32)
        conditions = jnp.array(self.condition_data[start_idx:start_idx + self.sequence_length], dtype=jnp.float32)
        return inputs, labels, conditions

def create_train_val_datasets(file_path, sequence_length=256, val_split=0.1):
    dataset = ControlDataset(file_path, sequence_length)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset = ControlDataset(file_path, sequence_length, False)
    val_dataset = ControlDataset(file_path, sequence_length, False)
    
    train_dataset.target_data = dataset.target_data[:train_size * sequence_length]
    train_dataset.condition_data = dataset.condition_data[:train_size * sequence_length]
    val_dataset.target_data = dataset.target_data[train_size * sequence_length:]
    val_dataset.condition_data = dataset.condition_data[train_size * sequence_length:]
    
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
        conditions = []
        for idx in batch_indices:
            x, y, c = dataset[idx]
            inputs.append(x)
            targets.append(y)
            conditions.append(c)
            
        yield jnp.stack(inputs), jnp.stack(targets), jnp.stack(conditions)

# ============= PART 2: MODEL =============

class AdaptiveRMSNorm(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x, condition):
        condition_projection = nn.Dense(self.hidden_dim)(condition)
        scale = nn.Dense(1)(condition_projection)
        scale = jax.nn.sigmoid(scale) * 2

        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x_normalized = x * jax.lax.rsqrt(variance + 1e-5)
        
        weight = self.param('weight', 
                          nn.initializers.ones, 
                          (self.hidden_dim,))
        
        return x_normalized * weight * scale

class TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, condition, training: bool = True):
        # Attention with adaptive norm
        norm1 = AdaptiveRMSNorm(self.hidden_dim)
        normed_x = norm1(x, condition)
        
        # Updated attention call
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate)
        
        # The new API uses inputs_q, inputs_kv
        attention_output = attention(
            inputs_q=normed_x,
            inputs_kv=normed_x,
            deterministic=not training)
        x = x + attention_output

        # MLP with adaptive norm
        norm2 = AdaptiveRMSNorm(self.hidden_dim)
        normed_x = norm2(x, condition)
        
        dense1 = nn.Dense(self.hidden_dim * 4)
        dense2 = nn.Dense(self.hidden_dim)
        
        mlp_output = dense1(normed_x)
        mlp_output = nn.gelu(mlp_output)
        mlp_output = nn.Dropout(
            rate=self.dropout_rate)(mlp_output, deterministic=not training)
        mlp_output = dense2(mlp_output)
        
        return x + mlp_output

class ControlTransformer(nn.Module):
    num_layers: int
    hidden_dim: int
    num_heads: int
    dropout_rate: float
    feature_dim: int

    @nn.compact
    def __call__(self, x, condition, training: bool = True):
        # Input projection
        x = nn.Dense(self.hidden_dim)(x)

        # Positional embedding
        position = self.param('pos_embedding',
                            nn.initializers.normal(stddev=0.02),
                            (256, self.hidden_dim))
        x = x + position[:x.shape[1]]

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate)(x, condition, training)

        # Output projection
        x = AdaptiveRMSNorm(self.hidden_dim)(x, condition)
        x = nn.Dense(self.feature_dim)(x)
        return x

# ============= PART 3: TRAINING =============

class TrainState(train_state.TrainState):
    dropout_rng: Any

def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    dropout_rng, param_rng = jax.random.split(rng)
    
    model = ControlTransformer(
        num_layers=config['num_blocks'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        dropout_rate=config['dropout_rate'],
        feature_dim=config['feature_dim'])
    
    # Create dummy inputs for initialization
    dummy_input = jnp.ones((1, config['seq_len'], config['feature_dim']))
    dummy_condition = jnp.ones((1, config['seq_len'], config['condition_dim']))
    
    variables = model.init(param_rng, dummy_input, dummy_condition, training=True)
    
    tx = optax.adam(config['learning_rate'])
    
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        dropout_rng=dropout_rng)

@jax.jit
def train_step(state, batch, dropout_rng):
    """Train for a single step."""
    inputs, targets, conditions = batch
    dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)

    def loss_fn(params):
        predictions = state.apply_fn(
            {'params': params},
            inputs,
            conditions,
            training=True,
            rngs={'dropout': dropout_rng})
        loss = jnp.mean((predictions - targets) ** 2)
        return loss, predictions

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, predictions), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    state = state.replace(dropout_rng=new_dropout_rng)
    
    metrics = {
        'loss': loss,
        'mse': jnp.mean((predictions - targets) ** 2, axis=(0, 1)),
        'mae': jnp.mean(jnp.abs(predictions - targets), axis=(0, 1))
    }
    
    return state, metrics

@jax.jit
def eval_step(state, batch):
    """Evaluate for a single step."""
    inputs, targets, conditions = batch
    predictions = state.apply_fn(
        {'params': state.params},
        inputs,
        conditions,
        training=False)
    
    metrics = {
        'loss': jnp.mean((predictions - targets) ** 2),
        'mse': jnp.mean((predictions - targets) ** 2, axis=(0, 1)),
        'mae': jnp.mean(jnp.abs(predictions - targets), axis=(0, 1))
    }
    
    return metrics

def train(config):
    # Initialize random numbers
    rng = jax.random.PRNGKey(config['seed'])
    rng, init_rng = jax.random.split(rng)

    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        config['data_path'], 
        config['seq_len'], 
        config['val_split']
    )
    
    # Add feature dimensions to config
    config['feature_dim'] = train_dataset.feature_dim
    config['condition_dim'] = train_dataset.condition_dim

    # Create train state
    state = create_train_state(init_rng, config)

    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Training
        train_metrics = []
        progress_bar = tqdm.tqdm(
            get_batches(train_dataset, config['batch_size']),
            total=len(train_dataset) // config['batch_size'],
            desc="Training",
            leave=True)

        for batch in progress_bar:
            state, metrics = train_step(state, batch, state.dropout_rng)
            train_metrics.append(metrics)
            progress_bar.set_postfix({'loss': f"{metrics['loss'].item():.4f}"})

        # Validation
        val_metrics = []
        for batch in get_batches(val_dataset, config['batch_size']):
            metrics = eval_step(state, batch)
            val_metrics.append(metrics)

        # Print metrics
        train_loss = np.mean([m['loss'] for m in train_metrics])
        val_loss = np.mean([m['loss'] for m in val_metrics])
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            checkpoints.save_checkpoint(
                ckpt_dir="checkpoints",
                target=state,
                step=epoch + 1,
                overwrite=True)

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
        'dropout_rate': 0.1,
        'save_every': 5,
        'seed': 42
    }
    
    train(config)
