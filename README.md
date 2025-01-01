# grad
A minimal automatic differentiation library for tensor operations

## Fundamental Operations (Implemented)
- MatMul
- Add
- Exp
- Log
- Reshape
- Permute

## Composite Operations (Implemented via Fundamentals)
1. Hadamard Product (Element-wise multiplication)
```c
a ⊙ b = exp(log(a) + log(b))
```

2. Reduce Sum
```c
// Using matrix multiplication with a vector of ones
reduce_sum(A, axis) = ones @ A  // for axis=0
reduce_sum(A, axis) = A @ ones  // for axis=1
```

## Required Operations for Transformer Decoder

1. Softmax

softmax(x) = exp(x - log(sum(exp(x))))

```c
// Can be implemented using existing operations:
softmax(x) = exp(x) / sum(exp(x))
           = exp(x - max(x)) / sum(exp(x - max(x)))  // for numerical stability

// Breaking it down:
a) max(x) ≈ log(sum(exp(x)))  // log-sum-exp trick
b) normalized = exp(x - max(x))
c) sum_normalized = reduce_sum(normalized, axis=-1)
d) result = normalized / sum_normalized
         = exp(log(normalized) - log(sum_normalized))
```

2. Layer Normalization
```c
// Can be composed from:
a) mean = reduce_sum(x, axis=-1) / n
b) variance = reduce_sum((x - mean)², axis=-1) / n
c) normalized = (x - mean) / sqrt(variance + ε)
d) result = γ * normalized + β

// Where sqrt(x) = exp(0.5 * log(x))
```

3. Masked Attention
a) Q @ K.T
b) mask addition
c) softmax
d) result @ V


4. Feed Forward Network
```c
// Just needs existing operations:
a) MatMul
b) Add (for bias)
c) ReLU or GELU
   // ReLU(x) = max(0,x) can be approximated with softplus
   // GELU(x) ≈ x * sigmoid(1.702 * x)
```