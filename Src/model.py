import torch
import torch.nn as nn
import torch.nn.functional as F

class BridgePolicyNet(nn.Module):
    """
    橋牌打牌策略網路 (Actor)
    
    輸入 165 維狀態向量：
      [0-51]   : 歷史出牌紀錄
      [52-103] : 當前出牌者手牌 (Legal Masking 作用區)
      [104-155]: 夢家手牌
      [156-160]: 王牌 one-hot (5 維)
      [161-164]: 順位 one-hot (4 維)
    
    輸出 52 維 logits（對應 52 張牌），非法牌自動設為 -1e4。
    
    架構改動：
      - 使用 LayerNorm 取代 BatchNorm1d，避免 eval() 模式下的 running stats 偏差
      - 移除 Dropout（RL 訓練不需要正則化防 overfitting，需要穩定的確定性輸出）
      - hidden_dim 從 1024 降至 512，配合 RL 推論速度需求
    """
    def __init__(self, input_dim=165, hidden_dim=512, output_dim=52):
        super(BridgePolicyNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # 取出手牌向量作為合法動作遮罩來源
        hand = x[:, 52:104]          # (B, 52)：手牌位元

        # 計算原始 logits
        logits = self.net(x)         # (B, 52)

        # --- 合法動作遮罩 (Legal Masking) ---
        # 手牌中沒有的牌 (hand == 0) 設為極小值，讓 softmax 後機率趨近 0
        mask = (hand == 0)
        logits = logits.masked_fill(mask, -1e4)

        return logits


# ============================================================
# ResBlock：殘差模塊（帶 skip connection，解決梯度消失）
# ============================================================
class ResBlock(nn.Module):
    """
    殘差模塊 (Residual Block)

    結構：
        輸入 x
          ├─→ [Linear → LayerNorm → ReLU → Linear → LayerNorm] → F(x)
          └─────────────────────────────────────────────────────→ x（跳接）
        輸出：ReLU(F(x) + x)

    數學：H(x) = F(x) + x
    梯度：∂H/∂x = ∂F/∂x + 1  ← 永遠有 +1，不會消失
    """
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))   # 殘差加法後 ReLU


# ============================================================
# BridgePolicyNetResNet：ResNet 版策略網路（供 RL 訓練使用）
# ============================================================
class BridgePolicyNetResNet(nn.Module):
    """
    橋牌打牌策略網路（ResNet 版）—— 供 RL 強化學習訓練使用

    架構（預設 4 個殘差模塊）：
        165 → [輸入投影 512] → [ResBlock×4] → [輸出頭 512→256→52]

    相比 BridgePolicyNet（MLP）的優勢：
        - 殘差連接讓梯度直通，可安全堆疊更多層
        - 每個 ResBlock 可選擇「學習改變」或「直接跳過」
        - 適合深度強化學習的長時間訓練

    注意：本類別與 BridgePolicyNet（監督學習用）完全獨立，
         不影響現有的監督預訓練模型。
    """
    def __init__(self, input_dim: int = 165, hidden_dim: int = 512,
                 output_dim: int = 52, num_blocks: int = 4):
        super().__init__()

        # 輸入投影：將 165 維特徵升維至 hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # 殘差模塊堆疊（num_blocks 個，預設 4 個）
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # 輸出頭：從 hidden_dim 映射到 52 張牌的 logits
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        # 取出手牌向量作為合法動作遮罩來源
        hand = x[:, 52:104]    # (B, 52)：手牌位元

        # 輸入投影
        h = self.input_proj(x)

        # 逐層通過殘差模塊
        for block in self.res_blocks:
            h = block(h)

        # 輸出頭
        logits = self.output_head(h)    # (B, 52)

        # --- 合法動作遮罩 (Legal Masking) ---
        mask = (hand == 0)
        logits = logits.masked_fill(mask, -1e4)

        return logits


# ============================================================
# LegacyPolicyNet：舊版監督預訓練策略網路
# ============================================================
class LegacyPolicyNet(nn.Module):
    """
    舊版監督預訓練策略網路，對齊 policy_165dim_best.pth 的架構。

    架構（fc_net key 對應關係）：
      fc_net.0  → Linear(165, 1024)
      fc_net.1  → BatchNorm1d(1024)
      fc_net.2  → ReLU
      fc_net.3  → Dropout(0.3)
      fc_net.4  → Linear(1024, 1024)
      fc_net.5  → ReLU
      fc_net.6  → Dropout(0.3)
      fc_net.7  → Linear(1024, 512)
      fc_net.8  → ReLU
      fc_net.9  → Linear(512, 52)

    與 BridgePolicyNet（新版）的差異：
      - 使用 BatchNorm1d（非 LayerNorm），hidden_dim=1024（非 512）
      - 包含 Dropout（RL 版已移除）
      - state_dict key 前綴為 fc_net，非 net
    """
    def __init__(self, input_dim: int = 165):
        super().__init__()
        self.fc_net = nn.Sequential(
            nn.Linear(input_dim, 1024),   # fc_net.0
            nn.BatchNorm1d(1024),          # fc_net.1
            nn.ReLU(),                     # fc_net.2
            nn.Dropout(0.3),               # fc_net.3
            nn.Linear(1024, 1024),         # fc_net.4
            nn.ReLU(),                     # fc_net.5
            nn.Dropout(0.3),               # fc_net.6
            nn.Linear(1024, 512),          # fc_net.7
            nn.ReLU(),                     # fc_net.8
            nn.Linear(512, 52),            # fc_net.9
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_net(x)