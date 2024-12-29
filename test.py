import torch

# Create tensors with the same data
a = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
b = torch.tensor([[5., 6.], [7., 8.]], requires_grad=True)

# First computation: d = (a + b) @ b
c = a + b
d = c @ b

print("Forward pass:")
print(f"a: {a.data}")
print(f"b: {b.data}")
print(f"c = a + b: {c.data}")
print(f"d = c @ b: {d.data}")

# Backward pass
d.backward(torch.tensor([[1., 0.], [0., 0.]]))

print("\nBackward pass:")
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")

# Reset gradients
a.grad.zero_()
b.grad.zero_()

# New computation: e = a @ b
e = a @ b
e.backward(torch.tensor([[1., 0.], [0., 0.]]))

print("\nNew computation:")
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")
print(f"e = a @ b: {e.data}")