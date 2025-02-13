# mlp
A multilayer perceptron implementation

Consider a standard feed-forward network operating on batched inputs of shape (batch_size × input_dim). The architecture consists of two linear transformations with an intermediate swish activation, where the forward propagation follows:

$$
\begin{align*}
Z &= XW_1 \\
A &= Z\sigma(Z) \\
Y &= AW_2
\end{align*}
$$

The swish activation $x\sigma(x)$ interpolates between linear and nonlinear regimes, yielding the following backward pass through the chain rule, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
\frac{\partial L}{\partial Y} &= Y - Y_{\text{true}} \\
\frac{\partial L}{\partial W_2} &= A^\top(\frac{\partial L}{\partial Y}) \\
\frac{\partial L}{\partial A} &= (\frac{\partial L}{\partial Y})(W_2)^\top \\
\frac{\partial L}{\partial Z} &= \frac{\partial L}{\partial A} \odot [\sigma(Z) + Z\sigma(Z)(1-\sigma(Z))] \\
\frac{\partial L}{\partial W_1} &= X^\top(\frac{\partial L}{\partial Z})
\end{align*}
$$

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```