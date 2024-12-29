import jax
import jax.numpy as jnp

def feed_forward(params, x):
    """Feed-forward network with GELU activation"""
    x = jax.numpy.dot(x, params['in_weight'])
    x1, x2 = jax.numpy.split(x, 2, axis=2)
    x = x1 * jax.nn.sigmoid(x2)  # Using SwiGLU activation
    x = jax.numpy.dot(x, params['out_weight'])
    return x

def attention(params, x, mask, batch_size, seq_len, num_heads, hidden_dim):
    """Multi-head attention mechanism"""
    # Linear transformations
    q = jax.numpy.dot(x, params['q_linear']).reshape(batch_size, seq_len, num_heads, (hidden_dim // num_heads)).transpose(0, 2, 1, 3)
    k = jax.numpy.dot(x, params['k_linear']).reshape(batch_size, seq_len, num_heads, (hidden_dim // num_heads)).transpose(0, 2, 1, 3)
    v = jax.numpy.dot(x, params['v_linear']).reshape(batch_size, seq_len, num_heads, (hidden_dim // num_heads)).transpose(0, 2, 1, 3)
    
    # Compute attention scores
    scores = jax.numpy.matmul(q, k.transpose(0, 1, 3, 2)) * ((hidden_dim // num_heads) ** -0.5)
    scores = jax.nn.softmax(scores + mask, axis=3)
    
    # Compute output
    output = jax.numpy.matmul(scores, v)
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_dim)
    output = jax.numpy.dot(output, params['o_linear'])
    return output

def transformer_block(params, x, mask, batch_size, seq_len, num_heads, hidden_dim):
    """Single transformer block"""
    x = x + attention(params['attention'], simple_rms_norm(x), mask, batch_size, seq_len, num_heads, hidden_dim)
    x = x + feed_forward(params['feed_forward'], simple_rms_norm(x))
    return x

def simple_rms_norm(x):
    """Root mean square normalization"""
    x = x * jax.lax.rsqrt(jax.numpy.mean(jax.numpy.square(x), axis=2, keepdims=True) + 1e-5)
    return x

def control_model(params, inputs, mask, batch_size, seq_len, num_heads, hidden_dim):
    """Main model for control data prediction"""
    # Project input features to hidden dimension
    x = jax.numpy.dot(inputs, params['input_projection'])
    
    # Add positional embeddings
    x = x + params['positional_embedding'][:seq_len]
    
    # Apply transformer blocks
    for block_params in params['transformer_blocks']:
        x = transformer_block(block_params, x, mask, batch_size, seq_len, num_heads, hidden_dim)
    
    # Project back to feature dimension
    x = jax.numpy.dot(simple_rms_norm(x), params['output_projection'])
    return x

def init_params(feature_dim, seq_len, num_blocks=8, num_heads=8, hidden_dim=512, ff_dim=2048, dtype=jnp.float32, rng_key=jax.random.key(0)):
    """Initialize model parameters"""
    xavier_uniform_init = jax.nn.initializers.glorot_uniform(dtype=dtype)
    kaiming_normal_init = jax.nn.initializers.he_normal(dtype=dtype)
    
    # Split RNG key for different parameter initializations
    rng_keys = jax.random.split(rng_key, num_blocks * 6 + 4)
    key_idx = 0
    
    learnable_params = {
        # Input and output projections
        'input_projection': xavier_uniform_init(rng_keys[key_idx], (feature_dim, hidden_dim)),
        'output_projection': xavier_uniform_init(rng_keys[key_idx + 1], (hidden_dim, feature_dim)),
        
        # Positional embedding
        'positional_embedding': jax.random.normal(rng_keys[key_idx + 2], (seq_len, hidden_dim)) * 0.02,
        
        # Transformer blocks
        'transformer_blocks': []
    }
    key_idx += 3
    
    # Initialize transformer blocks
    for _ in range(num_blocks):
        block_params = {
            'attention': {
                'q_linear': xavier_uniform_init(rng_keys[key_idx], (hidden_dim, hidden_dim)),
                'k_linear': xavier_uniform_init(rng_keys[key_idx + 1], (hidden_dim, hidden_dim)),
                'v_linear': xavier_uniform_init(rng_keys[key_idx + 2], (hidden_dim, hidden_dim)),
                'o_linear': xavier_uniform_init(rng_keys[key_idx + 3], (hidden_dim, hidden_dim)),
            },
            'feed_forward': {
                'in_weight': kaiming_normal_init(rng_keys[key_idx + 4], (hidden_dim, ff_dim)),
                'out_weight': xavier_uniform_init(rng_keys[key_idx + 5], (ff_dim // 2, hidden_dim)),
            }
        }
        learnable_params['transformer_blocks'].append(block_params)
        key_idx += 6
    
    # Create causal attention mask
    mask = jnp.tril(jnp.ones((seq_len, seq_len))) * -1e10
    mask = jnp.where(mask == 0, 0, mask)
    mask = jnp.broadcast_to(mask[None, None, :, :], (batch_size, num_heads, seq_len, seq_len))
    
    static_config = {
        "mask": mask,
        "num_heads": num_heads,
        "hidden_dim": hidden_dim
    }
    
    return learnable_params, static_config

def create_model(feature_dim, seq_len=256, batch_size=32, num_blocks=8, num_heads=8, hidden_dim=512, ff_dim=2048, dtype=jnp.float32, seed=0):
    """Convenience function to create model with default parameters"""
    rng_key = jax.random.key(seed)
    learnable_params, static_config = init_params(
        feature_dim=feature_dim,
        seq_len=seq_len,
        num_blocks=num_blocks,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        ff_dim=ff_dim,
        dtype=dtype,
        rng_key=rng_key
    )
    
    return {
        'learnable_params': learnable_params,
        'static_config': static_config,
        'model_config': {
            'feature_dim': feature_dim,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'num_blocks': num_blocks,
            'num_heads': num_heads,
            'hidden_dim': hidden_dim,
            'ff_dim': ff_dim,
        }
    }

if __name__ == "__main__":
    # Example usage
    feature_dim = 10  # Number of control features
    seq_len = 256
    batch_size = 32
    
    # Create model
    model = create_model(
        feature_dim=feature_dim,
        seq_len=seq_len,
        batch_size=batch_size,
        num_blocks=8,
        num_heads=8,
        hidden_dim=512,
        ff_dim=2048
    )
    
    # Create dummy input
    dummy_input = jnp.zeros((batch_size, seq_len, feature_dim))
    
    # Test forward pass
    output = control_model(
        model['learnable_params'],
        dummy_input,
        model['static_config']['mask'],
        batch_size,
        seq_len,
        model['static_config']['num_heads'],
        model['static_config']['hidden_dim']
    )
    
    print(f"Model output shape: {output.shape}")