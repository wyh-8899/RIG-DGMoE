import torch
import torch.nn as nn
import torch.nn.functional as F


def build_neighbor_padded(adj: torch.Tensor, add_self_loop: bool = True, k_max: int = None):
    """
    adj: [N,N] (0/1 or weight)
    return:
      nb_idx:  [N,K] long  (padding with 0)
      nb_mask: [N,K] bool
    """
    assert adj.dim() == 2 and adj.size(0) == adj.size(1)
    N = adj.size(0)

    A = adj.clone()
    if add_self_loop:
        A = torch.maximum(A, torch.eye(N, device=A.device, dtype=A.dtype))

    # neighbors list
    nb_list = []
    max_deg = 0
    for i in range(N):
        nei = torch.nonzero(A[i] > 0, as_tuple=False).squeeze(-1)  # [deg]
        nb_list.append(nei)
        max_deg = max(max_deg, int(nei.numel()))

    K = max_deg if (k_max is None) else min(max_deg, int(k_max))
    nb_idx = torch.zeros((N, K), dtype=torch.long, device=A.device)
    nb_mask = torch.zeros((N, K), dtype=torch.bool, device=A.device)

    for i in range(N):
        nei = nb_list[i]
        if nei.numel() == 0:
            # 理论上不会发生（add_self_loop=True），兜底：指向自己
            nei = torch.tensor([i], device=A.device, dtype=torch.long)

        if nei.numel() > K:
            # 这里简单截断；更高级可以按权重/度排序后取前K
            nei = nei[:K]

        nb_idx[i, :nei.numel()] = nei
        nb_mask[i, :nei.numel()] = True

    return nb_idx, nb_mask


class TransitionNet(nn.Module):
    """
    ✅ 输出邻接候选概率 P_nb: [B,N,K]
    - 输入 x: [B,N,in_dim]
    - 邻接 adj: [N,N] (静态) 用于构造 nb_idx/nb_mask（只构造一次，作为 buffer）
    """
    def __init__(self, in_dim: int, hidden_dim: int, adj_mx: torch.Tensor,
                 dropout: float = 0.1, add_self_loop: bool = True, k_max: int = None):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # --- build neighbor list once ---
        if not torch.is_tensor(adj_mx):
            adj_mx = torch.tensor(adj_mx)
        nb_idx, nb_mask = build_neighbor_padded(adj_mx, add_self_loop=add_self_loop, k_max=k_max)
        self.register_buffer("nb_idx", nb_idx)     # [N,K]
        self.register_buffer("nb_mask", nb_mask)   # [N,K]

    def forward(self, x: torch.Tensor):
        """
        Return:
          P_nb:   [B,N,K]  row-stochastic over neighbors
          nb_idx: [N,K]
          nb_mask:[N,K]
        """
        B, N, _ = x.shape
        K = self.nb_idx.size(1)

        h = self.dropout(self.activation(self.ln1(self.fc1(x))))
        h = self.dropout(self.activation(self.ln2(self.fc2(h))))  # [B,N,H]

        # h: [B, N, H]
        # nb_idx: [N, K]
        h_nb = h[:, self.nb_idx, :]          # ✅ [B, N, K, H]
        score = (h.unsqueeze(2) * h_nb).sum(dim=-1)   # [B, N, K]
        score = score.masked_fill(~self.nb_mask.unsqueeze(0), -1e9)
        P_nb = F.softmax(score, dim=-1)      # [B, N, K]
        return P_nb, self.nb_idx, self.nb_mask