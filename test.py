import torch

# 假设 A 和 B 的形状是 (1101, 1101, 8) 和 (1101, 16, 8)
A = torch.randn(1101, 1101, 8)
B = torch.randn(1101, 16, 8)

# 转置 B 使其形状为 (1101, 8, 16)
B_transposed = B.permute(0, 2, 1)

# 执行批量矩阵乘法
result = torch.matmul(A, B_transposed)

# 对结果的最后一维进行求和，得到形状为 (1101, 8)
result_summed = result.sum(dim=1)

# 输出结果的形状
print(result_summed.shape)  # 应该是 (1101, 8)