import torch
import torch.nn as nn

class BridgePolicyNet(nn.Module):
    def __init__(self, input_dim=165, hidden_dim=1024, output_dim=52):
        super(BridgePolicyNet, self).__init__()
        
        self.fc_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
    # 0-51: 歷史紀錄 (52維)
    # 52-103: 當前出牌者手牌 (52維) -> Masking 作用區
    # 104-155: 夢家手牌 (52維) -> 這是我們新增的情報！
    # 156-164: 環境變數 (王牌 5 維 + 順位 4 維)
        
        hand = x[:, 52:104] 
        
        # 計算原始輸出
        logits = self.fc_net(x)
        
        # --- 合法動作遮罩 (Legal Masking) ---
        mask = (hand == 0)
        logits = logits.masked_fill(mask, -1e4)
        
        return logits