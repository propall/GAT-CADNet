import torch

# Assume that the shapes of A and B are (1101, 1101, 8) and (1101, 16, 8)
A = torch.randn(1101, 1101, 8)
B = torch.randn(1101, 16, 8)

# Transpose B to shape it (1101, 8, 16)
B_transposed = B.permute(0, 2, 1)

# Perform batch matrix multiplication
result = torch.matmul(A, B_transposed)

# Summarize the last dimension of the result and get the shape (1101, 8)
result_summed = result.sum(dim=1)

# The shape of the output result
print(result_summed.shape)  # Probably (1101, 8)