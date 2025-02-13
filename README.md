# mlp
A multilayer perceptron implementation

Consider a standard feed-forward network operating on batched inputs of shape (batch_size × input_dim). The architecture consists of two linear transformations with an intermediate swish activation, where the forward propagation follows:

```
Z = XW₁
A = Zσ(Z)    
Y = AW₂       
```

The swish activation xσ(x) interpolates between linear and nonlinear regimes, yielding the following backward pass through the chain rule, where ⊙ denotes elementwise multiplication:

```
∂L/∂Y = Y - Y_true
∂L/∂W₂ = Aᵀ(∂L/∂Y)
∂L/∂A = (∂L/∂Y)(W₂)ᵀ
∂L/∂Z = ∂L/∂A ⊙ [σ(Z) + Zσ(Z)(1-σ(Z))]
∂L/∂W₁ = Xᵀ(∂L/∂Z)
```

The AdamW optimizer maintains exponential moving averages of gradients and their squares through β₁ and β₂, while simultaneously applying L2 regularization through weight decay λ. The learning rate is denoted by η, t is the current training iteration, and ε is a small constant for numerical stability. For each weight matrix W, the update rule is:

```
m = β₁m + (1-β₁)(∂L/∂W)
v = β₂v + (1-β₂)(∂L/∂W)²
W = (1-λη)W - η·(m/(1-β₁ᵗ))/√(v/(1-β₂ᵗ) + ε)
```

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```