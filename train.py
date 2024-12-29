import jax
import tqdm
import random
import numpy as np
from datetime import datetime
from data import create_train_val_datasets
from model import create_model, control_model
from optim import create_adam_state, apply_adam_optimizer

def compute_metrics(predictions, targets):
    """Compute MSE and MAE for each feature"""
    mse = jax.numpy.mean(jax.numpy.square(predictions - targets), axis=(0, 1))
    mae = jax.numpy.mean(jax.numpy.abs(predictions - targets), axis=(0, 1))
    return {'mse': mse, 'mae': mae}

def loss_fn(learnable_params, inputs, targets, mask):
    """Compute MSE loss with static shapes"""
    predictions = control_model(
        learnable_params, 
        inputs, 
        mask,
        batch_size=32,  # Static
        seq_len=256,    # Static
        num_heads=8,    # Static
        hidden_dim=384  # Static
    )
    loss = jax.numpy.mean(jax.numpy.square(predictions - targets))
    return loss, predictions

@jax.jit
def train_step(learnable_params, optimizer_state, inputs, targets, mask):
    """Single training step with static shapes"""
    (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        learnable_params, inputs, targets, mask
    )
    learnable_params, optimizer_state = apply_adam_optimizer(learnable_params, optimizer_state, grads)
    metrics = compute_metrics(predictions, targets)
    return learnable_params, optimizer_state, loss, metrics

@jax.jit
def eval_step(learnable_params, inputs, targets, mask):
    """Single evaluation step with static shapes"""
    loss, predictions = loss_fn(learnable_params, inputs, targets, mask)
    metrics = compute_metrics(predictions, targets)
    return loss, metrics

def prepare_batch_data(dataset, indices, batch_size):
    """Prepare batch data"""
    if len(indices) < batch_size:
        return None, None
    
    inputs = []
    targets = []
    
    for idx in indices[:batch_size]:
        x, y = dataset[idx]
        inputs.append(x)
        targets.append(y)
    
    # Stack into batch
    inputs = jax.numpy.stack(inputs)
    targets = jax.numpy.stack(targets)
    
    return inputs, targets

def train(config):
    """Main training loop"""
    print("\nInitializing training...")
    
    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        config['data_path'],
        sequence_length=config['seq_len'],
        val_split=config['val_split'],
        scale_data=True
    )

    # Create model
    print("\nCreating model...")
    model = create_model(
        feature_dim=train_dataset.feature_dim,
        seq_len=config['seq_len'],
        batch_size=config['batch_size'],
        num_blocks=config['num_blocks'],
        num_heads=config['num_heads'],
        hidden_dim=config['hidden_dim'],
        ff_dim=config['ff_dim'],
        seed=config['seed']
    )

    # Initialize optimizer
    optimizer_state = create_adam_state(
        model['learnable_params'],
        learning_rate=config['learning_rate']
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    step = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Training
        train_losses = []
        train_metrics = {'mse': [], 'mae': []}
        
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)
        
        with tqdm.tqdm(range(0, len(indices), config['batch_size'])) as pbar:
            for i in pbar:
                # Prepare batch
                batch_indices = indices[i:i + config['batch_size']]
                inputs, targets = prepare_batch_data(train_dataset, batch_indices, config['batch_size'])
                
                if inputs is None:
                    continue
                
                # Training step
                model['learnable_params'], optimizer_state, loss, metrics = train_step(
                    model['learnable_params'],
                    optimizer_state,
                    inputs,
                    targets,
                    model['static_config']['mask']
                )
                
                train_losses.append(loss)
                train_metrics['mse'].append(metrics['mse'])
                train_metrics['mae'].append(metrics['mae'])
                
                # Update progress bar
                pbar.set_description(f"Training - Loss: {loss:.6f}")
                
                step += 1
        
        # Print training metrics
        epoch_train_loss = np.mean(train_losses)
        epoch_train_mse = np.mean([m for m in train_metrics['mse']], axis=0)
        epoch_train_mae = np.mean([m for m in train_metrics['mae']], axis=0)
        
        print(f"\nTraining Metrics:")
        print(f"Loss: {epoch_train_loss:.6f}")
        print(f"MSE: {epoch_train_mse.mean():.6f}")
        print(f"MAE: {epoch_train_mae.mean():.6f}")
        
        # Validation
        print("\nRunning validation...")
        val_losses = []
        val_metrics = {'mse': [], 'mae': []}
        
        for i in range(0, len(val_dataset), config['batch_size']):
            inputs, targets = prepare_batch_data(
                val_dataset,
                list(range(i, min(i + config['batch_size'], len(val_dataset)))),
                config['batch_size']
            )
            
            if inputs is None:
                continue
            
            loss, metrics = eval_step(
                model['learnable_params'],
                inputs,
                targets,
                model['static_config']['mask']
            )
            
            val_losses.append(loss)
            val_metrics['mse'].append(metrics['mse'])
            val_metrics['mae'].append(metrics['mae'])
        
        # Print validation metrics
        epoch_val_loss = np.mean(val_losses)
        epoch_val_mse = np.mean([m for m in val_metrics['mse']], axis=0)
        epoch_val_mae = np.mean([m for m in val_metrics['mae']], axis=0)
        
        print(f"\nValidation Metrics:")
        print(f"Loss: {epoch_val_loss:.6f}")
        print(f"MSE: {epoch_val_mse.mean():.6f}")
        print(f"MAE: {epoch_val_mae.mean():.6f}")
        
        # Save checkpoint if best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            checkpoint_path = f"checkpoint_{step}_{epoch_val_loss:.6f}.npz"
            jax.numpy.savez(
                checkpoint_path,
                learnable_params=model['learnable_params'],
                static_config_mask=model['static_config']['mask'],
                static_config_num_heads=model['static_config']['num_heads'],
                static_config_hidden_dim=model['static_config']['hidden_dim'],
                model_config=model['model_config']
            )
            print(f"\nSaved new best model: {checkpoint_path}")

if __name__ == "__main__":
    # Training configuration
    config = {
        # Data
        'data_path': '2024-12-29_6-25-1_control_data.csv',
        'val_split': 0.1,
        
        # Model (smaller for faster training)
        'seq_len': 256,
        'batch_size': 32,
        'num_blocks': 6,  # Reduced from 8
        'num_heads': 8,
        'hidden_dim': 384,  # Reduced from 512
        'ff_dim': 1536,  # Reduced from 2048
        
        # Training
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'seed': random.randint(0, 2**16-1),
    }
    
    print("Training configuration:")
    for k, v in config.items():
        print(f"{k}: {v}")
    
    # Set random seed
    random.seed(config['seed'])
    
    # Start training
    train(config)