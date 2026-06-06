# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
from gymnasium import spaces
import numpy as np
import random
from Src.utils import determine_trick_winner, CARD_TO_IDX

# =========================
# 基本常數
# =========================
SUITS = ['S', 'H', 'D', 'C']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
PLAYERS = ['N', 'E', 'S', 'W'] # 對應 index 0, 1, 2, 3

class BridgeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        
        self.NUM_CARDS = 52

        # 動作空間：0~51 代表出牌的 index
        self.action_space = spaces.Discrete(self.NUM_CARDS)

        # 狀態空間：165維，與 dataset.py 完全對齊
        # [歷史出牌 52, 自己手牌 52, 夢家手牌 52, 王牌 5, 順位 4]
        # 同時使用 Dict 來包含 action_mask，方便 RL 模型避開非法動作
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1.0, shape=(165,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(52,), dtype=np.int8)
        })

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        deck = list(range(52)) # 0~51
        self.np_random.shuffle(deck)

        # 發牌給四個玩家
        self.hands = {
            0: deck[0:13],
            1: deck[13:26],
            2: deck[26:39],
            3: deck[39:52]
        }
        
        # 隨機決定莊家 (Declarer)，以跳過叫牌階段
        self.declarer_idx = self.np_random.integers(0, 4)
        self.dummy_idx = (self.declarer_idx + 2) % 4
        # 首攻者是莊家的左手邊
        self.opening_leader_idx = (self.declarer_idx + 1) % 4
        self.current_player_idx = self.opening_leader_idx
        
        # 隨機決定王牌 0:S, 1:H, 2:D, 3:C, 4:NT
        self.trump_idx = self.np_random.integers(0, 5)
        self.trump_vec = np.zeros(5, dtype=np.float32)
        self.trump_vec[self.trump_idx] = 1.0

        # 初始化遊戲狀態
        self.game_over = False
        self.current_trick_cards = []
        self.current_trick_players = []
        self.history = np.zeros(52, dtype=np.int8)
        self.trick_count = 0
        
        self.tricks_won = {
            'NS': 0,
            'EW': 0
        }

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        curr_p = self.current_player_idx
        
        # 1. 歷史出牌 (52維)
        history_vec = self.history.astype(np.float32)
        
        # 2. 自己手牌 (52維)
        hand_vec = np.zeros(52, dtype=np.float32)
        for card_idx in self.hands[curr_p]:
            hand_vec[card_idx] = 1.0
            
        # 3. 夢家手牌 (52維)
        dummy_vec = np.zeros(52, dtype=np.float32)
        # 防止資訊洩漏：全域首攻時 (第一磴的第一張牌) 夢家手牌為隱藏 (全 0)
        # 一旦打出第一張牌，夢家就攤牌公開
        if self.trick_count > 0 or len(self.current_trick_cards) > 0:
            for card_idx in self.hands[self.dummy_idx]:
                dummy_vec[card_idx] = 1.0
                
        # 4. 順位特徵 (4維)
        pos_vec = np.zeros(4, dtype=np.float32)
        pos_idx = len(self.current_trick_cards)
        pos_vec[pos_idx] = 1.0
        
        obs = np.concatenate([
            history_vec,
            hand_vec,
            dummy_vec,
            self.trump_vec,
            pos_vec
        ])
        
        return {
            "observation": obs,
            "action_mask": self.get_legal_actions()
        }

    def _get_info(self):
        return {
            "current_player": self.current_player_idx,
            "declarer": self.declarer_idx,
            "trump_idx": self.trump_idx,
            "tricks_won": self.tricks_won
        }

    def get_legal_actions(self):
        """產生合法出牌的 Mask"""
        mask = np.zeros(52, dtype=np.int8)
        curr_p = self.current_player_idx
        hand = self.hands[curr_p]
        
        if len(self.current_trick_cards) == 0:
            # 領出 (Lead)：手上的牌都可以出
            for card in hand:
                mask[card] = 1
        else:
            # 跟出 (Follow)：必須跟隨首引花色
            lead_card = self.current_trick_cards[0]
            lead_suit = lead_card // 13
            
            has_lead_suit = any((c // 13) == lead_suit for c in hand)
            
            for card in hand:
                card_suit = card // 13
                if has_lead_suit:
                    if card_suit == lead_suit:
                        mask[card] = 1
                else:
                    # 缺門 (Void)：本來有的牌都可以墊牌或王吃
                    mask[card] = 1
                    
        return mask

    def step(self, action):
        if self.game_over:
            return self._get_obs(), 0.0, True, False, self._get_info()

        curr_p = self.current_player_idx
        
        # 1. 檢查動作合法性
        legal_actions = self.get_legal_actions()
        if legal_actions[action] == 0:
            # 模型嘗試了不合法的出牌
            info = self._get_info()
            info["error"] = "illegal_card"
            
            # 給予違規玩家嚴厲的負回報，其他玩家 0
            rewards = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
            rewards[curr_p] = -1.0 
            info["rewards"] = rewards
            
            # 提早結束這局
            return self._get_obs(), -1.0, True, False, info

        # 2. 執行出牌
        self.hands[curr_p].remove(action)
        self.history[action] = 1
        self.current_trick_cards.append(action)
        self.current_trick_players.append(curr_p)
        
        reward = 0.0
        
        # 3. 判斷磴與贏家
        if len(self.current_trick_cards) == 4:
            # 使用我們之前修復好的判斷函式
            winner_idx = determine_trick_winner(
                self.current_trick_cards, 
                self.current_trick_players, 
                self.trump_idx
            )
            
            # 紀錄該隊得分
            if winner_idx in [0, 2]: # N 或 S 贏得這磴
                self.tricks_won['NS'] += 1
            else: # E 或 W 贏得這磴
                self.tricks_won['EW'] += 1
                
            self.trick_count += 1
            # 贏家取得下一磴的首攻
            self.current_player_idx = winner_idx 
            
            self.current_trick_cards = []
            self.current_trick_players = []
            
            # 13磴打完，遊戲結束
            if self.trick_count == 13:
                self.game_over = True
        else:
            # 還沒滿 4 張，換順時針下一位玩家
            self.current_player_idx = (self.current_player_idx + 1) % 4
            
        info = self._get_info()
        
        # 4. 結算最終分數 (Multi-Agent Rewards)
        if self.game_over:
            ns_won = self.tricks_won['NS']
            ew_won = self.tricks_won['EW']
            
            # 由於我們跳過了叫牌，這裡採用相對吃磴數計算回報
            # 13 磴中，吃到過半 (6.5) 就為正，少於過半就為負，歸一化至 [-1, 1] 區間
            ns_reward = (ns_won - 6.5) / 6.5 
            ew_reward = (ew_won - 6.5) / 6.5
            
            # 將四個代理人的回報打包在 info 中
            # 自行設計的 PyTorch 訓練迴圈可以透過 info["rewards"] 抓出對應玩家的分數去算 Loss
            info["rewards"] = {
                0: ns_reward,
                2: ns_reward,
                1: ew_reward,
                3: ew_reward
            }
            
            # 為了符合 Gym 格式，step 的主要 reward 回傳給剛剛動作的玩家
            reward = info["rewards"][curr_p]

        return self._get_obs(), reward, self.game_over, False, info

    def render(self):
        print(f"\n--- Trick {self.trick_count+1} ---")
        print(f"Trump: {['S', 'H', 'D', 'C', 'NT'][self.trump_idx]}")
        print(f"Declarer: {PLAYERS[self.declarer_idx]}")
        print(f"Current Player: {PLAYERS[self.current_player_idx]}")
        print(f"Score NS: {self.tricks_won['NS']} | EW: {self.tricks_won['EW']}")
        
        played_cards_str = [f"{SUITS[c // 13]}{RANKS[c % 13]}" for c in self.current_trick_cards]
        print(f"Cards played this trick: {played_cards_str}")
        
    def close(self):
        pass

if __name__ == "__main__":
    env = BridgeEnv()
    obs, info = env.reset()
    done = False
    
    print("=== 隨機對弈測試開始 ===")
    # 渲染第一磴一開始的狀態 (顯示 Trick 1)
    env.render()
    
    while not done:
        # RL 模型訓練時，請利用這個 action_mask 過濾輸出
        mask = obs["action_mask"]
        legal_actions = np.where(mask == 1)[0]
        
        # 隨機挑選合法動作
        action = np.random.choice(legal_actions)
        
        curr = info["current_player"]
        # 將 card index 轉換成撲克牌字串，如 S2, HA
        card_str = f"{SUITS[action // 13]}{RANKS[action % 13]}"
        print(f"Player {PLAYERS[curr]} plays {card_str}")
        
        obs, reward, done, truncated, info = env.step(action)
        
        # 當一磴打完 (current_trick_cards 被清空)，印出贏家與下一磴的開始
        if len(env.current_trick_cards) == 0 and not done:
            winner = PLAYERS[info["current_player"]]
            print(f">>> 本磴由 Player {winner} 贏得！\n")
            env.render()
            
    print("\n=== 遊戲結束 ===")
    print("各隊最終得分 NS:", env.tricks_won['NS'], "| EW:", env.tricks_won['EW'])
    print("各代理人最終 Rewards:", {PLAYERS[k]: round(v, 2) for k, v in info.get("rewards").items()})