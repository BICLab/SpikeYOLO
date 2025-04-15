import torch
import torch.nn.functional as F

def expand_tensor_cumulative(tensor, max_value=4):

    T, B, C, H, W = tensor.shape
    # 创建一个 shape 为 [max_value, 1, 1, 1, 1, 1] 的比较向量
    steps = torch.arange(max_value, device=tensor.device).view(max_value, 1, 1, 1, 1, 1)

    # 扩展原始张量维度，便于比较 → [1, T, B, C, H, W]
    tensor_expanded = tensor.unsqueeze(0)

    # 比较：每个位置 v，生成 v 个 1，其余为 0
    binary = (steps < tensor_expanded).float()  # → shape [max_value, T, B, C, H, W]

    # 重新 reshape → [max_value * T, B, C, H, W]
    binary = binary.permute(1, 0, 2, 3, 4, 5).reshape(T * max_value, B, C, H, W)

    return binary

if __name__ == "__main__":
    x = torch.randint(0, 6, (2, 3, 1, 4, 4))  # [T=2, B=3, C=1, H=4, W=4]
    y = expand_tensor_cumulative(x, max_value=5)

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("x sum:", x.sum(0))
    print("y sum:", y.sum(0))