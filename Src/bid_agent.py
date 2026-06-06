"""
BidAgent - 叫牌代理

負責：
  1. 載入叫牌模型 (BiddingModel)
  2. 根據觀察與合法動作選出最佳叫牌
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

# 將專案根目錄加入 path，以便 import policy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from Src.bidding_model import BiddingModel


class BidAgent:
    """叫牌代理：封裝模型載入與動作選擇"""

    def __init__(self, model_path, device=None):
        """
        參數:
            model_path: 叫牌模型權重檔案路徑
            device: torch 設備 (預設自動選擇 cuda/cpu)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiddingModel().to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到叫牌模型檔案: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"[BidAgent] 叫牌模型載入成功 (設備: {self.device})")

    def select_action(self, bidding_state, legal_actions):
        """
        根據 78 維叫牌觀察向量與合法動作列表，選出最佳叫牌。

        參數:
            bidding_state: np.ndarray, shape=(78,) — 環境產生的叫牌特徵
            legal_actions: list[int] — 合法叫牌動作索引 (0~37)

        回傳:
            action: int — 選中的叫牌動作索引
            confidence: float — 該動作的機率 (0~1)
        """
        state_tensor = torch.FloatTensor(bidding_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(state_tensor)

            # 合法動作遮罩
            mask = torch.full((38,), -1e9, device=self.device)
            for a in legal_actions:
                mask[a] = 0
            logits = logits + mask

            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        action = int(np.argmax(probs))
        confidence = float(probs[action])
        return action, confidence
