import torch
import torch.nn.functional as F
import numpy as np
import os
from Src.model import BridgePolicyNet
from Src.utils import CARD_TO_IDX, IDX_TO_CARD, get_legal_mask

def run_session():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "Data/models/policy_165dim_best.pth" # 建議讀取你剛剛存的最佳版本
    
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型 {model_path}")
        return

    model = BridgePolicyNet(input_dim=165).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("\n" + "="*40)
    print("       橋牌 AI 連續對局助手        ")
    print("="*40)

    # --- 1. 初始化對局狀態 ---
    print("\n[初始化] 請輸入原始 13 張手牌:")
    hand_input = input(">> ").upper().split()
    current_hand = [CARD_TO_IDX.get(c) for c in hand_input if CARD_TO_IDX.get(c) is not None]
    
    # 🌟 新增：取得夢家情報
    print("\n[初始化] 請輸入夢家的 13 張牌 (若尚在叫牌階段不知夢家，請直接 Enter):")
    dummy_input = input(">> ").upper().split()
    dummy_hand = [CARD_TO_IDX.get(c) for c in dummy_input if CARD_TO_IDX.get(c) is not None]

    print("\n[初始化] 請輸入合約王牌 (S, H, D, C, NT):")
    t_input = input(">> ").upper()
    t_map = {'S':0, 'H':1, 'D':2, 'C':3, 'NT':4}
    t_idx = t_map.get(t_input, 4)

    full_history = []

    # --- 2. 進入對局循環 ---
    while len(current_hand) > 0:
        print("\n" + "-"*40)
        print(f"目前剩餘手牌: {[IDX_TO_CARD[i] for i in sorted(current_hand)]}")
        if len(dummy_hand) > 0:
            print(f"目前夢家手牌: {[IDX_TO_CARD[i] for i in sorted(dummy_hand)]}")
        
        print(f"\n[輸入] 請輸入「這一輪」新增的歷史牌張 (若你是首攻請直接 Enter):")
        new_hist = input(">> ").upper().split()
        
        lead_card_idx = None
        if len(new_hist) > 0:
            lead_card_idx = CARD_TO_IDX.get(new_hist[0])
        
        for c in new_hist:
            idx = CARD_TO_IDX.get(c)
            if idx is not None: 
                full_history.append(idx)
                # 如果這張牌是夢家出的，要從夢家手牌中扣除
                if idx in dummy_hand:
                    dummy_hand.remove(idx)

        print(f"[輸入] 你的出牌順位 (1, 2, 3, 4):")
        try:
            pos_input = int(input(">> "))
        except:
            pos_input = 1

        # --- 3. 🚀 組成 165 維向量 (正確對齊版) ---
        state = np.zeros(165, dtype=np.float32)
        
        # [0-51] 已出牌歷史
        for idx in full_history: 
            state[idx] = 1.0        
            
        # [52-103] 當前手牌
        hand_vec = np.zeros(52, dtype=np.int8)
        for idx in current_hand: 
            state[52 + idx] = 1.0 
            hand_vec[idx] = 1
            
        # [104-155] 夢家手牌
        for idx in dummy_hand:
            state[104 + idx] = 1.0

        # [156-160] 王牌
        state[156 + t_idx] = 1.0 
        
        # [161-164] 順位
        state[161 + (pos_input - 1)] = 1.0

        # --- 4. 推理 (加上規則過濾) ---
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(state_tensor) # [1, 52]
            
            legal_mask = get_legal_mask(hand_vec, lead_card_idx)
            torch_mask = torch.from_numpy(~legal_mask).to(device).unsqueeze(0)
            logits = logits.masked_fill(torch_mask, -1e9)
            
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # --- 5. 輸出建議 ---
        recommendations = []
        for idx in current_hand:
            if probs[idx] > 1e-6:
                recommendations.append((idx, probs[idx] * 100))
        recommendations.sort(key=lambda x: x[1], reverse=True)

        print("\n--- AI 建議 (已過濾非法花色) ---")
        for i, (idx, p) in enumerate(recommendations[:5]):
            print(f"{i+1}. {IDX_TO_CARD[idx]}: {p:6.2f}% " + "█" * int(p/2))

        # --- 6. 狀態更新 ---
        print(f"\n[決策] 你最後決定出哪一張牌？ (輸入牌名，例如 {IDX_TO_CARD[recommendations[0][0]]})")
        played_card = input(">> ").upper()
        played_idx = CARD_TO_IDX.get(played_card)
        
        if played_idx in current_hand:
            current_hand.remove(played_idx)
            full_history.append(played_idx)
            print(f" 已從手牌移除 {played_card}，並加入歷史紀錄。")
        else:
            print(" 警告：輸入的牌不在手牌中，狀態未更新。")

        if input("\n繼續下一輪？ (y/n): ").lower() != 'y': break

    print("\n對局結束！")

if __name__ == "__main__":
    run_session()