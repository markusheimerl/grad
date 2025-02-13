# mlp
A multilayer perceptron implementation

Consider a standard feed-forward network operating on batched input X ∈ ℝᵇˣⁱ. The architecture consists of two linear transformations with an intermediate swish activation, where the forward propagation follows:

```
Z¹ = XW¹
A¹ = Z¹σ(Z¹)    
Y  = A¹W²       
```

The swish activation xσ(x) interpolates between linear and nonlinear regimes, yielding the following backward pass through the chain rule:

```
∂L/∂Y = Y - Y_true
∂L/∂W² = (A¹)ᵀ(∂L/∂Y)
∂L/∂A¹ = (∂L/∂Y)(W²)ᵀ
∂L/∂Z¹ = ∂L/∂A¹ ⊙ [σ(Z¹) + Z¹σ(Z¹)(1-σ(Z¹))]
∂L/∂W¹ = Xᵀ(∂L/∂Z¹)
```

The AdamW optimizer maintains exponential moving averages of gradients and their squares through β₁ and β₂, while simultaneously applying L2 regularization through weight decay λ:

```
m_t = β₁m_{t-1} + (1-β₁)∇W
v_t = β₂v_{t-1} + (1-β₂)(∇W)²
W_t = (1-λη)W_{t-1} - η·m̂_t/√(v̂_t + ε)
```

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```