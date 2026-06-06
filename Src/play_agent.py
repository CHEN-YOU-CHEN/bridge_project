"""
PlayAgent - 出牌代理

負責：
  1. 載入出牌模型 (BridgePolicyNet)
  2. 根據觀察與合法牌組選出最佳出牌
"""

import torch
import torch.nn.functional as F
import numpy as np
import os

from Src.model import BridgePolicyNet
from Src.bridge_gym_env import IDX_TO_CARD


class PlayAgent:
    """出牌代理：封裝模型載入與動作選擇"""

    def __init__(self, model_path, device=None):
        """
        參數:
            model_path: 出牌模型權重檔案路徑
            device: torch 設備 (預設自動選擇 cuda/cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BridgePolicyNet(input_dim=165).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到出牌模型檔案: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[PlayAgent] 出牌模型載入成功 (設備: {self.device})")

    def select_action(self, playing_state, legal_card_indices):
        """
        根據 165 維打牌觀察向量與合法卡牌索引，選出最佳出牌。

        參數:
            playing_state: np.ndarray, shape=(165,) — 環境產生的打牌特徵
            legal_card_indices: list[int] — 合法出牌的卡牌索引 (0~51)

        回傳:
            card_idx: int — 選中的卡牌索引 (0~51)
            card_name: str — 卡牌名稱 (如 'AS', 'KH')
            confidence: float — 該動作的機率 (0~1)
        """
        state_tensor = torch.FloatTensor(playing_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(state_tensor)

            # 合法動作遮罩
            mask = torch.full((52,), -1e9, device=self.device)
            for idx in legal_card_indices:
                mask[idx] = 0
            logits = logits + mask

            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        card_idx = int(np.argmax(probs))
        card_name = IDX_TO_CARD[card_idx]
        confidence = float(probs[card_idx])
        return card_idx, card_name, confidence
