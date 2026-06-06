import numpy as np

# --- 基礎定義 ---
SUITS = ['S', 'H', 'D', 'C']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
TRUMP_MAP = {'S': 0, 'H': 1, 'D': 2, 'C': 3, 'N': 4} # 4 代表 NT (無王)

# 建立 52 張牌的對照表 (例如: 'SA' -> 12, 'H5' -> 16)
CARD_TO_IDX = {f"{s}{r}": i * 13 + j for i, s in enumerate(SUITS) for j, r in enumerate(RANKS)}
IDX_TO_CARD = {i * 13 + j: f"{s}{r}" for i, s in enumerate(SUITS) for j, r in enumerate(RANKS)}

def card_str_to_idx(card_str):
    """將 BBO 格式字串 (如 'HA') 轉為 0-51 索引"""
    if not card_str or len(card_str) < 2:
        return None
    # 統一轉為大寫並處理可能的順序問題 (BBO 有時會寫 AH 有時 HA)
    s = card_str[0].upper()
    r = card_str[1].upper()
    
    # 修正 BBO 縮寫格式 (有時花色在後，需做簡單判斷)
    if s not in SUITS and r in SUITS:
        s, r = r, s
        
    key = f"{s}{r}"
    return CARD_TO_IDX.get(key)

def get_suit_from_idx(idx):
    """根據索引回傳花色 (0:S, 1:H, 2:D, 3:C)"""
    return idx // 13

def get_trump_vec(game_str):
    """從遊戲內容中找出王牌並轉為 5維 One-hot，同時回傳 trump_idx"""
    vec = np.zeros(5, dtype=np.float32)
    trump_idx = 4 # 預設 NT
    
    # 1. 嘗試從 tr| 直接讀取
    if 'tr|' in game_str:
        t = game_str.split('tr|')[1][0].upper()
        trump_idx = TRUMP_MAP.get(t, 4)
    else:
        # 2. 從 mb| 序列中解析最後一個有效叫牌
        parts = game_str.split('|')
        last_bid = None
        for i in range(len(parts)):
            if parts[i] == 'mb' and i + 1 < len(parts):
                bid = parts[i+1].upper().strip()
                if bid and bid not in ['P', 'D', 'R'] and bid[0].isdigit():
                    last_bid = bid
        
        if last_bid:
            # last_bid 例如 "3N" 或 "4S"
            suit_char = last_bid[-1]
            trump_idx = TRUMP_MAP.get(suit_char, 4)
            
    vec[trump_idx] = 1.0
    return vec, trump_idx

# --- 核心邏輯：合法動作判定 ---

def get_legal_mask(hand_vec, lead_card_idx=None):
    """
    hand_vec: 52維 0/1 向量 (目前玩家的手牌)
    lead_card_idx: int (本輪第一張出的牌的索引), 如果是首攻則為 None
    """
    # 預設：手上有的牌都可以打
    mask = hand_vec.copy().astype(bool)
    
    if lead_card_idx is None:
        return mask
    
    # 判斷領出的花色 (0:S, 1:H, 2:D, 3:C)
    lead_suit = lead_card_idx // 13
    start, end = lead_suit * 13, (lead_suit + 1) * 13
    
    # 檢查玩家手上有沒有該花色
    has_lead_suit = np.any(hand_vec[start:end] > 0)
    
    if has_lead_suit:
        # 強制跟花色：將非該花色的區域全部設為 False
        final_mask = np.zeros(52, dtype=bool)
        final_mask[start:end] = hand_vec[start:end].astype(bool)
        return final_mask
    
    # 如果沒該花色，本來有的牌(mask)都可以出
    return mask

def determine_trick_winner(cards, players, trump_idx):
    """
    決定一磴的贏家。
    cards: list of 4 card indices (0-51)
    players: list of 4 player indices (0-3)
    trump_idx: 0:S, 1:H, 2:D, 3:C, 4:NT
    """
    if len(cards) != 4 or len(players) != 4:
        return players[0]

    lead_suit = cards[0] // 13
    best_card = cards[0]
    best_player = players[0]
    
    for c, p in zip(cards[1:], players[1:]):
        c_suit = c // 13
        c_rank = c % 13
        
        best_suit = best_card // 13
        best_rank = best_card % 13
        
        is_trump_c = (trump_idx != 4 and c_suit == trump_idx)
        is_trump_best = (trump_idx != 4 and best_suit == trump_idx)
        
        if is_trump_c and not is_trump_best:
            best_card = c
            best_player = p
        elif is_trump_c and is_trump_best:
            if c_rank > best_rank:
                best_card = c
                best_player = p
        else:
            if not is_trump_best and c_suit == lead_suit:
                if c_rank > best_rank:
                    best_card = c
                    best_player = p
                
    return best_player

# --- 輔助功能：State 向量化輔助 ---

def encode_contract(contract_str):
    """
    將合約字串轉為向量 (例如 '4S' -> [0, 1, 0, 0, 0] 其中一位代表黑桃)
    這裡可根據研究需求定義，通常包含王牌花色(5種，含NT)與階數
    """
    # 簡化版：回傳王牌花色索引 (0:S, 1:H, 2:D, 3:C, 4:NT)
    trump_map = {'S': 0, 'H': 1, 'D': 2, 'C': 3, 'N': 4}
    for char in contract_str.upper():
        if char in trump_map:
            return trump_map[char]
    return 4 # 預設無王