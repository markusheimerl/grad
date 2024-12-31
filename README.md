# grad
A differentiation library for efficient tensor operations

ToDo:
- Training support on GitHub runners
- Artifact generation and package releases
- Reduction operations (reduce_sum, reduce_mean, reduce_max, reduce_min)
- Inverse Square root operation (1/√x)
- Softmax operation across dimension (composable)

Done:
- ✓ MatMul
- ✓ Add
- ✓ Relu
- ✓ Sigmoid
- ✓ Reshape
- ✓ Slice
- ✓ Permute
- ✓ Gather
- ✓ Hadamard
- ✓ Pow

Note:
Only ADD,EXP,LOG,MATMUL and RESHAPE seem fundamental...

1. HADAMARD (as we already saw):
```c
// a ⊙ b = exp(log(a) + log(b))
```

2. DIV (division):
```c
// a / b = exp(log(a) - log(b))
// where subtraction is ADD with negative
```

3. POW:
```c
// a^n = exp(n * log(a))
// where multiplication by scalar can be done via diagonal matrix multiplication
```

4. REDUCE_SUM:
```c
// Can be implemented via MATMUL with a vector of ones
// For example, to sum a matrix along rows:
// [1 1 1] @ [a b c] = [a+b+c]
//           [d e f]   [d+e+f]
```

5. SIGMOID:
```c
// sigmoid(x) = 1 / (1 + exp(-x))
// = exp(-log(1 + exp(-x)))
// Can be composed using EXP, LOG, and ADD
```

6. REDUCE_MAX:
```c
// Can be approximated using log-sum-exp trick:
// max(x) ≈ log(sum(exp(x)))