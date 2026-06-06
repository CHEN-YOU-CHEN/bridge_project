import torch
import numpy as np
import random
import os

# --- 基礎還原邏輯 ---
SUITS = ['S', 'H', 'D', 'C']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

def idx_to_card_str(idx):
    """將 0-51 索引轉回牌名 (如 12 -> SA)"""
    if idx < 0 or idx >= 52: return "???"
    suit = SUITS[idx // 13]
    rank = RANKS[idx % 13]
    return f"{suit}{rank}"

def check_pt_file(file_path, num_samples=3):
    if not os.path.exists(file_path):
        print(f"錯誤：找不到檔案 {file_path}")
        return

    print(f"\n{'='*50}")
    print(f"🔍 開始檢查資料集: {file_path}")
    print(f"{'='*50}")
    
    data = torch.load(file_path)
    states = data['states']
    actions = data['actions']
    
    total = len(actions)
    dim = states.shape[1]
    print(f" 總樣本數: {total} | 特徵維度: {dim}\n")

    # 隨機抽樣
    indices = random.sample(range(total), num_samples)
    
    for count, idx in enumerate(indices, 1):
        s = states[idx].numpy()
        a = int(actions[idx])
        
        # 1. 拆解基礎特徵 (前 104 維是固定的)
        history_vec = s[0:52]
        hand_vec = s[52:104]
        
        my_hand = [idx_to_card_str(i) for i, val in enumerate(hand_vec) if val > 0]
        history = [idx_to_card_str(i) for i, val in enumerate(history_vec) if val > 0]
        action_card = idx_to_card_str(a)
        
        print(f"▶ [抽樣 {count} | 索引 {idx}]")
        print(f"   手牌 ({len(my_hand)}張): {my_hand}")
        
        # 2. 如果是 165 維，解析夢家特徵
        if dim == 165:
            dummy_vec = s[104:156]
            dummy_hand = [idx_to_card_str(i) for i, val in enumerate(dummy_vec) if val > 0]
            print(f"   夢家 ({len(dummy_hand)}張): {dummy_hand}")
            
            # 特殊檢查：如果手牌與夢家完全一致，代表這是夢家的回合
            if np.array_equal(hand_vec, dummy_vec):
                print("     (系統提示：本回合由莊家操作【夢家】出牌)")
            else:
                dummy_overlap = np.logical_and(hand_vec, dummy_vec).any()
                print(f"     (檢查: 手牌與夢家重疊? {' 異常' if dummy_overlap else ' 正常(獨立)'})")

        print(f"   歷史 ({len(history)}張): {history}")
        print(f"   專家動作: {action_card}")
        
        # 3. 核心合法性檢查
        is_legal = hand_vec[a] > 0
        overlap = np.logical_and(history_vec, hand_vec).any()
        
        print(f"  狀態檢查: {'合法 (手裡有這張牌)' if is_legal else '錯誤 (手裡沒這張牌)'}")
        print(f"  歷史檢查: {'異常 (歷史與手牌重疊)' if overlap else '正常 (已出牌不在手裡)'}")

        # 4. 解析環境變數 (位置取決於維度)
        trump_vec, pos_vec = None, None
        if dim == 113:
            trump_vec = s[104:109]
            pos_vec = s[109:113]
        elif dim == 165:
            trump_vec = s[156:161]
            pos_vec = s[161:165]

        if trump_vec is not None and pos_vec is not None:
            trumps = ['S', 'H', 'D', 'C', 'NT']
            trump_suit = trumps[np.argmax(trump_vec)]
            pos_num = np.argmax(pos_vec) + 1
            print(f"  環境資訊: 王牌={trump_suit}, 出牌順位=第 {pos_num} 家")
            
        print("-" * 50)

if __name__ == "__main__":
    path = "Data/processed/bridge_dataset.pt"
    check_pt_file(path, num_samples=5)