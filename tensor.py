import torch

B, T, C = 2, 4, 8

x = torch.randn(B, T, C)

print("x:", x.shape)
print("x[:, 0, :]:", x[:, 0, :].shape)
print("x[0, :, :]:", x[0, :, :].shape)
print("x[:, :, 0]:", x[:, :, 0].shape)

linear = torch.nn.Linear(C, C)
print("linear(x):", linear(x).shape)

embedding = torch.nn.Embedding(10000, C)

input_ids = torch.tensor([
    [12, 45, 78, 91],
    [3, 17, 29, 100]
])

print("embedding(input_ids):", embedding(input_ids).shape)