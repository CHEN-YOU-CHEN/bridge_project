import os
import torch
import numpy as np
from torch.utils.data import Dataset
from Src.utils import card_str_to_idx

class BridgeDataset(Dataset):
    def __init__(self, raw_data_dir=None, processed_path=None):
        self.states = None
        self.actions = None
        if processed_path and os.path.exists(processed_path):
            print(f"載入預處理資料: {processed_path}")
            data = torch.load(processed_path)
            self.states, self.actions = data['states'], data['actions']
        elif raw_data_dir:
            print(f"開始從 {raw_data_dir} 處理原始 LIN 檔案...")
            self._process_raw_files(raw_data_dir)
            if processed_path: 
                self.save(processed_path)
        else:
            print("請提供路徑。")

    def _parse_bbo_hand(self, hand_str):
        """將 BBO 手牌字串 (如 SAJ974H65...) 轉為 52維 0/1 向量"""
        vec = np.zeros(52, dtype=np.int8)
        suit_letters = {'S': 0, 'H': 1, 'D': 2, 'C': 3}
        rank_map = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,'9':7,'T':8,'J':9,'Q':10,'K':11,'A':12}
        
        current_suit = -1
        for char in hand_str.upper():
            if char in suit_letters:
                current_suit = suit_letters[char]
            elif char in rank_map and current_suit != -1:
                idx = current_suit * 13 + rank_map[char]
                vec[idx] = 1
        return vec

    def _process_raw_files(self, data_dir):
        file_list = [f for f in os.listdir(data_dir) if f.endswith('.lin')]
        temp_states, temp_actions = [], []
        
        for f_idx, file_name in enumerate(file_list):
            try:
                with open(os.path.join(data_dir, file_name), 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for game in content.split('qx|'):
                    if 'md|' not in game or 'pc|' not in game: continue
                    
                    # A. 解析王牌 (5維)
                    from Src.utils import TRUMP_MAP, get_trump_vec, determine_trick_winner
                    trump_vec, trump_idx = get_trump_vec(game)

                    # B. 解析發牌與補全手牌 (維持原樣)
                    md_part = game.split('md|')[1].split('|')[0]
                    dealer_pos = int(md_part[0]) - 1 
                    hand_groups = md_part[1:].split(',')
                    player_hands = [None] * 4
                    all_cards_mask = np.zeros(52, dtype=np.int8)
                    for i in range(len(hand_groups)):
                        p_idx = (dealer_pos + i) % 4
                        h_vec = self._parse_bbo_hand(hand_groups[i])
                        player_hands[p_idx] = h_vec
                        all_cards_mask |= h_vec
                    for i in range(4):
                        if player_hands[i] is None: player_hands[i] = 1 - all_cards_mask

                    # --- 🚀 C. 找出首攻者與夢家位置 ---
                    parts = game.split('|')
                    play_seq = [card_str_to_idx(parts[i+1]) for i in range(len(parts)) if parts[i]=='pc' and card_str_to_idx(parts[i+1]) is not None]
                    
                    if len(play_seq) == 0: continue
                    
                    # 透過第一張出牌，反推誰是首攻者
                    first_card = play_seq[0]
                    opening_leader = -1
                    for p in range(4):
                        if player_hands[p][first_card] == 1:
                            opening_leader = p
                            break
                    
                    if opening_leader == -1: continue # 資料異常則跳過

                    # 夢家固定在首攻者的左手邊
                    dummy_pos = (opening_leader + 1) % 4
                    # 複製一份夢家的手牌作為獨立特徵
                    dummy_hand_vec = player_hands[dummy_pos].copy()

                    # 遊戲從首攻者開始
                    current_player = opening_leader 
                    history = np.zeros(52, dtype=np.int8)
                    
                    current_trick_cards = []
                    current_trick_players = []

                    # --- D. 模擬對局並生成 165 維特徵 ---
                    for i, action in enumerate(play_seq):
                        if player_hands[current_player][action] == 0:
                            break 
                        
                        # 順位特徵 (4維 One-hot)
                        pos_vec = np.zeros(4, dtype=np.float32)
                        pos_vec[i % 4] = 1.0

                        # 防止資訊洩漏：全域首攻 (i==0) 時，夢家手牌尚未攤開，應視為全未知 (0)
                        visible_dummy = np.zeros(52, dtype=np.float32) if i == 0 else dummy_hand_vec.astype(np.float32)

                        # 組合 165 維: [歷史52, 自己手牌52, 夢家手牌52, 王牌5, 順位4]
                        state = np.concatenate([
                            history.astype(np.float32), 
                            player_hands[current_player].astype(np.float32),
                            visible_dummy, # 修正夢家特徵
                            trump_vec,
                            pos_vec
                        ])
                        
                        temp_states.append(state)
                        temp_actions.append(action)

                        # 更新狀態
                        history[action] = 1
                        player_hands[current_player][action] = 0
                        # 安全扣除：如果這張牌是夢家出的，把它從夢家向量中扣掉
                        dummy_hand_vec[action] = 0 
                        
                        current_trick_cards.append(action)
                        current_trick_players.append(current_player)
                        
                        # 換下一家出牌
                        if len(current_trick_cards) == 4:
                            current_player = determine_trick_winner(current_trick_cards, current_trick_players, trump_idx)
                            current_trick_cards = []
                            current_trick_players = []
                        else:
                            current_player = (current_player + 1) % 4

            except Exception as e: 
                # 可以根據需要取消註解以查看錯誤原因
                # print(f"Error parsing file {file_name}: {e}")
                continue

        print("轉換為大型張量並儲存中...")
        self.states = torch.from_numpy(np.array(temp_states))
        self.actions = torch.from_numpy(np.array(temp_actions, dtype=np.int64))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'states': self.states, 'actions': self.actions}, path)
        print(f"預處理完成！資料已儲存至: {path}")

    def __len__(self):
        return len(self.actions) if self.actions is not None else 0

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

if __name__ == "__main__":
    raw_dir = "Data/bbo_data"
    save_path = "Data/processed/bridge_dataset.pt"
    
    dataset = BridgeDataset(raw_data_dir=raw_dir, processed_path=save_path)
    print(f"最終有效訓練樣本總數: {len(dataset)}")