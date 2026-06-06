import torch
import torch.nn.functional as F
import numpy as np
import os

# --- 引入環境與模型 ---
from Src.bridge_env import BridgeEnv, BIDDING_ACTIONS
from Src.model import BridgePolicyNet       # 打牌模型
from Src.bidding_model import BiddingModel  # 叫牌模型 

# --- 準備對應表 ---
SUITS_TRAIN = ['S', 'H', 'D', 'C'] 
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
PLAYERS = ['N', 'E', 'S', 'W']

CARD_TO_IDX_ENV = {}
for suit_idx, suit in enumerate(SUITS_TRAIN):
    for rank_idx, rank in enumerate(RANKS):
        idx = suit_idx * 13 + rank_idx
        CARD_TO_IDX_ENV[f"{rank}{suit}"] = idx


def env_state_to_bidding_dim(env):
    """將環境狀態翻譯為叫牌模型所需的向量 (52 + 20 = 72維)"""
    
    # 1. 取得手牌 (52維)
    hand_vec = np.zeros(52, dtype=np.float32)
    current_player = env.get_current_player()
    for card_str in env.hands[current_player]:
        hand_vec[CARD_TO_IDX_ENV[card_str]] = 1.0
        
    # 2. 歷史叫牌 (20維)
    history_len = 20
    hist = env.bidding_history[-history_len:] 
    
    pad_len = history_len - len(hist)
    history_vec = np.array([-1] * pad_len + hist, dtype=np.float32)
    
    # 組合為 72 維
    state = np.concatenate([hand_vec, history_vec])
    return state

def env_state_to_165dim(env, full_history):
    """將環境狀態翻譯為打牌模型所需的 165 維向量"""
    state = np.zeros(165, dtype=np.float32)
    for card_str in full_history: state[CARD_TO_IDX_ENV[card_str]] = 1.0
    current_player = env.get_current_player()
    for card_str in env.hands[current_player]: state[52 + CARD_TO_IDX_ENV[card_str]] = 1.0
    declarer_idx = PLAYERS.index(env.declarer)
    dummy_player = PLAYERS[(declarer_idx + 2) % 4]
    for card_str in env.hands[dummy_player]: state[104 + CARD_TO_IDX_ENV[card_str]] = 1.0
    contract_suit = env.contract[-1] if env.contract[-2:] != 'NT' else 'NT'
    t_map = {'S':0, 'H':1, 'D':2, 'C':3, 'NT':4}
    state[156 + t_map.get(contract_suit, 4)] = 1.0
    state[161 + len(env.current_trick)] = 1.0
    return state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 啟動雙核心 AI 代理人 (使用設備: {device}) ===")

    # --- 1. 載入打牌模型 (Play) ---
    play_model_path = "Data/models/policy_165dim_best.pth" 
    play_model = BridgePolicyNet(input_dim=165).to(device)
    play_model.load_state_dict(torch.load(play_model_path, map_location=device))
    play_model.eval()
    
    # --- 2. 載入叫牌模型 (Bid) ---
    bid_model_path = "Data/models/bidding_model.pth"
    bid_model = BiddingModel(history_len=20, num_bid_types=39).to(device)
    if os.path.exists(bid_model_path):
        bid_model.load_state_dict(torch.load(bid_model_path, map_location=device))
        bid_model.eval()
        print("打牌與叫牌模型皆載入成功！")
    else:
        print(f"找不到叫牌模型 {bid_model_path}")
        return

    # --- 初始化環境 ---
    env = BridgeEnv()
    state = env.reset()
    print("\n=== 發牌結果 ===")
    for p in PLAYERS: print(f"{p} 手牌: {env.hands[p]}")


    print("\n=== 開始 AI 叫牌階段 ===")
    while not state['playing_phase'] and not env.game_over:
        current_player = state['player']
        legal_actions = env.get_legal_bidding_actions()
        
        # 1. 翻譯狀態
        state_vec = env_state_to_bidding_dim(env)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
        
        # 2. 推理與遮罩
        with torch.no_grad():
            logits = bid_model(state_tensor) # 輸出 38 維
            
            mask = torch.ones(38, dtype=torch.bool).to(device)
            for a in legal_actions:
                if a < 38: mask[a] = False 
                
            logits = logits.masked_fill(mask, -1e9)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        # 3. 挑選機率最高且合法的叫牌
        best_action = int(np.argmax(probs))
        action_name = BIDDING_ACTIONS[best_action]
        
        print(f"[{current_player}] AI 叫牌: {action_name:<4} (信心: {probs[best_action]*100:5.1f}%)")
        state, _, _, _ = env.step(best_action)

    print(f"\n叫牌結束 -> 最終合約: {state['contract']} | 莊家: {state['declarer']}")
    if state['contract'] == "Passed Out": return


    print("\n=== 開始 AI 打牌階段 ===")
    trick_count = 1
    full_history = [] 
    
    while not env.game_over:
        if len(env.current_trick) == 0:
            print(f"-- 第 {trick_count} 磴開始 -- (目前比分 NS:{state['tricks_won']['NS']} EW:{state['tricks_won']['EW']})")
            
        current_player = state['player']
        legal_cards = env.get_legal_playing_actions()
        
        state_vec = env_state_to_165dim(env, full_history)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = play_model(state_tensor)
            mask = torch.ones(52, dtype=torch.bool).to(device)
            for card in legal_cards: mask[CARD_TO_IDX_ENV[card]] = False 
            logits = logits.masked_fill(mask, -1e9)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
        best_card_idx = np.argmax(probs)
        card_to_play = [k for k, v in CARD_TO_IDX_ENV.items() if v == best_card_idx][0]
        
        print(f"[{current_player}] 出牌: {card_to_play} (AI 信心: {probs[best_card_idx]*100:5.1f}%)")
        
        full_history.append(card_to_play)
        state, _, done, _ = env.step(card_to_play)
        
        if len(env.current_trick) == 0 and not done:
            trick_count += 1
            print()

    print(f"\n=== 遊戲結束 ===\nNS 陣營: {state['tricks_won']['NS']} 磴 | EW 陣營: {state['tricks_won']['EW']} 磴")

if __name__ == "__main__":
    main()