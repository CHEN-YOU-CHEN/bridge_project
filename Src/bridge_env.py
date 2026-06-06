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


# =============================================================================
# 叫牌動作表
# 索引 0      : Pass
# 索引 1~35   : 1C, 1D, 1H, 1S, 1N, 2C, ..., 7N  (7階 × 5花色 = 35 種)
# 索引 36     : Double (加倍)
# 索引 37     : Redouble (再加倍)
# =============================================================================
_BID_SUITS = ['C', 'D', 'H', 'S', 'N']   # 按叫牌強度排序
BIDDING_ACTIONS = ['Pass'] + [
    f"{level}{suit}"
    for level in range(1, 8)
    for suit in _BID_SUITS
] + ['Double', 'Redouble']
# BIDDING_ACTIONS[0]  = 'Pass'
# BIDDING_ACTIONS[1]  = '1C' ... BIDDING_ACTIONS[35] = '7N'
# BIDDING_ACTIONS[36] = 'Double'
# BIDDING_ACTIONS[37] = 'Redouble'

# 將叫牌字串映射為 (level, suit_idx)，方便比較大小
_BID_TO_LEVEL_SUIT = {}
for _i, _b in enumerate(BIDDING_ACTIONS[1:36], start=1):
    _lv = int(_b[0])
    _su = _BID_SUITS.index(_b[1])
    _BID_TO_LEVEL_SUIT[_i] = (_lv, _su)


def _bid_strength(action_idx: int) -> int:
    """回傳叫牌強度值（越大越高），Pass/X/XX 不在比較範圍內回傳 -1"""
    if action_idx in _BID_TO_LEVEL_SUIT:
        lv, su = _BID_TO_LEVEL_SUIT[action_idx]
        return (lv - 1) * 5 + su   # 0~34
    return -1


# =============================================================================
# BridgeGymEnv
# 支援 train_playing_rl.py 的完整兩階段環境
# =============================================================================
class BridgeGymEnv:
    """
    橋牌兩階段 Gym 環境，提供叫牌 (Bidding) 與打牌 (Playing) 兩個子階段。

    叫牌觀察 (bidding_state): 72 維 np.float32
        [手牌 52 維] + [歷史叫牌 20 維 (用 -1 padding，0~37 為有效動作)]

    打牌觀察 (playing_state): 165 維 np.float32
        [歷史出牌 52] + [自己手牌 52] + [夢家手牌 52] + [王牌 5] + [順位 4]

    主要 API (對應 train_playing_rl.py)
    ─────────────────────────────────
    env.reset()                        -> (obs, info)
    env.playing_phase                  -> bool
    env.game_over                      -> bool
    env.contract                       -> str  (e.g. "3NT", "Passed Out")
    env.declarer                       -> str  (e.g. 'N', 'E', 'S', 'W')

    env.get_legal_bidding_actions()    -> list[int]
    env.step_bidding(action)           -> (obs, reward, terminated, truncated, info)

    env.get_current_player()           -> str  ('N'/'E'/'S'/'W')
    env.get_legal_playing_card_indices()-> list[int]  (0~51)
    env.step_playing(card_idx)         -> (obs, reward, terminated, truncated, info)

    env.tricks_won                     -> dict {'NS': int, 'EW': int}
    """

    # 最大叫牌歷史長度（與 BiddingModel 相同）
    MAX_BID_HISTORY = 20

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # --- 發牌 ---
        deck = list(range(52))
        random.shuffle(deck)
        self.hands = {
            'N': deck[0:13],
            'E': deck[13:26],
            'S': deck[26:39],
            'W': deck[39:52],
        }

        # --- 叫牌狀態 ---
        # 發牌者 (Dealer) 隨機決定
        self._dealer_idx      = random.randint(0, 3)   # 0=N,1=E,2=S,3=W
        self._bid_turn_idx    = self._dealer_idx       # 目前叫牌者
        self._bid_history     = []                     # list[int] 記錄實際叫牌動作索引
        self._last_real_bid   = 0                      # 上一個有效叫牌 (非 Pass/X/XX) 的動作索引
        self._last_real_bidder_idx = None              # 誰叫出了最後一個有效叫牌
        self._consecutive_pass = 0                     # 連續 Pass 計數
        self._doubled         = False                  # 是否已加倍
        self._redoubled       = False                  # 是否已再加倍
        self.playing_phase    = False
        self.game_over        = False

        # --- 打牌狀態 ---
        self.contract         = None   # str, e.g. "3NT"
        self.declarer         = None   # str, e.g. 'N'
        self._dummy           = None   # str, 夢家方位
        self._trump_idx       = None   # int 0:S,1:H,2:D,3:C,4:NT
        self._trump_vec       = None   # np.array (5,)
        self._opening_leader  = None   # str, 首攻者方位
        self._current_player  = None  # str, 目前出牌者
        self._history_vec     = np.zeros(52, dtype=np.float32)
        self._current_trick_cards   = []
        self._current_trick_players = []
        self.tricks_won       = {'NS': 0, 'EW': 0}
        self._trick_count     = 0

        obs  = self._get_obs()
        info = self._get_info()
        return obs, info

    # ------------------------------------------------------------------
    # 觀察 / 資訊
    # ------------------------------------------------------------------
    def _get_obs(self):
        if not self.playing_phase:
            return {
                'bidding_state':  self._make_bidding_state(),
                'playing_state':  np.zeros(165, dtype=np.float32),
                'action_mask':    None,
            }
        else:
            return {
                'bidding_state':  np.zeros(72, dtype=np.float32),
                'playing_state':  self._make_playing_state(),
                'action_mask':    self._make_play_mask(),
            }

    def _make_bidding_state(self) -> np.ndarray:
        """72 維叫牌觀察：[手牌 52] + [叫牌歷史最近 20 步, -1 padding]"""
        player = PLAYERS[self._bid_turn_idx]
        hand_vec = np.zeros(52, dtype=np.float32)
        for c in self.hands[player]:
            hand_vec[c] = 1.0

        hist = self._bid_history[-self.MAX_BID_HISTORY:]
        padded = [-1] * (self.MAX_BID_HISTORY - len(hist)) + hist
        hist_vec = np.array(padded, dtype=np.float32)

        return np.concatenate([hand_vec, hist_vec])

    def _make_playing_state(self) -> np.ndarray:
        """165 維打牌觀察：[歷史52] + [自己手牌52] + [夢家手牌52] + [王牌5] + [順位4]"""
        curr = self._current_player

        hand_vec = np.zeros(52, dtype=np.float32)
        for c in self.hands[curr]:
            hand_vec[c] = 1.0

        # 首磴第一張牌出之前，夢家手牌隱藏
        dummy_vec = np.zeros(52, dtype=np.float32)
        if self._trick_count > 0 or len(self._current_trick_cards) > 0:
            for c in self.hands[self._dummy]:
                dummy_vec[c] = 1.0

        pos_vec = np.zeros(4, dtype=np.float32)
        pos_vec[len(self._current_trick_cards)] = 1.0

        return np.concatenate([
            self._history_vec,
            hand_vec,
            dummy_vec,
            self._trump_vec,
            pos_vec,
        ])

    def _make_play_mask(self) -> np.ndarray:
        """回傳 52 維 0/1 mask"""
        mask = np.zeros(52, dtype=np.int8)
        curr = self._current_player
        hand = self.hands[curr]

        if len(self._current_trick_cards) == 0:
            for c in hand:
                mask[c] = 1
        else:
            lead_suit = self._current_trick_cards[0] // 13
            has_suit  = any(c // 13 == lead_suit for c in hand)
            for c in hand:
                if has_suit:
                    if c // 13 == lead_suit:
                        mask[c] = 1
                else:
                    mask[c] = 1
        return mask

    def _get_info(self):
        return {
            'phase':          'playing' if self.playing_phase else 'bidding',
            'current_player': PLAYERS[self._bid_turn_idx] if not self.playing_phase else self._current_player,
            'contract':       self.contract,
            'declarer':       self.declarer,
            'tricks_won':     dict(self.tricks_won),
            'game_over':      self.game_over,
        }

    # ------------------------------------------------------------------
    # 公開方法：目前玩家
    # ------------------------------------------------------------------
    def get_current_player(self) -> str:
        """回傳目前出牌者的方位字串 ('N'/'E'/'S'/'W')"""
        if self.playing_phase:
            return self._current_player
        return PLAYERS[self._bid_turn_idx]

    # ------------------------------------------------------------------
    # 叫牌階段
    # ------------------------------------------------------------------
    def get_legal_bidding_actions(self) -> list:
        """
        回傳目前合法的叫牌動作索引列表。
        規則：
          - Pass (0) 永遠合法
          - 有效叫牌必須高於目前最高叫牌
          - Double (36)：只有在對手叫了牌（且未加倍）時才合法
          - Redouble (37)：只有在我方被加倍時才合法
        """
        legal = [0]  # Pass 永遠合法

        # 有效叫牌：必須強度大於目前最高叫牌
        current_strength = _bid_strength(self._last_real_bid)
        for idx in range(1, 36):
            if _bid_strength(idx) > current_strength:
                legal.append(idx)

        # Double/Redouble 規則
        if self._last_real_bid > 0 and not self._doubled:
            # 對手方叫出了最後一個有效叫牌
            if self._last_real_bidder_idx is not None:
                if self._last_real_bidder_idx % 2 != self._bid_turn_idx % 2:
                    legal.append(36)  # Double

        if self._doubled and not self._redoubled:
            # 我方被加倍，可以再加倍
            if self._last_real_bidder_idx is not None:
                if self._last_real_bidder_idx % 2 != self._bid_turn_idx % 2:
                    legal.append(37)  # Redouble

        return legal

    def step_bidding(self, action: int):
        """
        執行一步叫牌。
        回傳 (obs, reward, terminated, truncated, info)
        叫牌階段 reward 恆為 0，terminated=True 代表叫牌結束（進入打牌或流局）
        """
        assert not self.playing_phase, "目前是打牌階段，請使用 step_playing()"
        assert not self.game_over, "遊戲已結束"

        self._bid_history.append(action)

        if action == 0:  # Pass
            self._consecutive_pass += 1
        elif action == 36:  # Double
            self._doubled = True
            self._consecutive_pass = 0
        elif action == 37:  # Redouble
            self._redoubled = True
            self._consecutive_pass = 0
        else:  # 有效叫牌
            self._last_real_bid = action
            self._last_real_bidder_idx = self._bid_turn_idx
            self._doubled   = False
            self._redoubled = False
            self._consecutive_pass = 0

        # 判斷叫牌是否結束
        bidding_done = False

        # 規則1: 四家都 Pass -> 流局
        if len(self._bid_history) >= 4 and all(a == 0 for a in self._bid_history[-4:]):
            bidding_done = True
            self.contract  = "Passed Out"
            self.game_over = True

        # 規則2: 有有效叫牌後，連續三個 Pass -> 叫牌結束
        elif self._last_real_bid > 0 and self._consecutive_pass >= 3:
            bidding_done = True
            self._resolve_contract()

        if not bidding_done:
            self._bid_turn_idx = (self._bid_turn_idx + 1) % 4

        obs  = self._get_obs()
        info = self._get_info()
        return obs, 0.0, bidding_done, False, info

    def _resolve_contract(self):
        """從最後有效叫牌確定合約、莊家、夢家、王牌、首攻者"""
        bid_str = BIDDING_ACTIONS[self._last_real_bid]   # e.g. '3N', '4S'
        level   = int(bid_str[0])
        suit_ch = bid_str[1]   # 'C','D','H','S','N'

        # 王牌索引  S=0, H=1, D=2, C=3, N=4（NT）
        _TRUMP_IDX = {'S': 0, 'H': 1, 'D': 2, 'C': 3, 'N': 4}
        self._trump_idx = _TRUMP_IDX[suit_ch]
        self._trump_vec = np.zeros(5, dtype=np.float32)
        self._trump_vec[self._trump_idx] = 1.0

        # 合約字串
        suffix_map = {'C': 'C', 'D': 'D', 'H': 'H', 'S': 'S', 'N': 'NT'}
        self.contract = f"{level}{suffix_map[suit_ch]}"

        # 莊家：叫出最後有效叫牌的人，或其合作夥伴中最早叫出該花色的人
        # 簡化：直接用最後叫牌者所在隊伍中，最早叫出該花色的玩家
        bidder_team_parity = self._last_real_bidder_idx % 2   # 0=NS, 1=EW
        self.declarer = self._find_declarer(bidder_team_parity, self._last_real_bid)

        declarer_idx    = PLAYERS.index(self.declarer)
        self._dummy     = PLAYERS[(declarer_idx + 2) % 4]     # 夢家=對家
        opening_leader_idx  = (declarer_idx + 1) % 4          # 首攻=莊家左手邊
        self._opening_leader = PLAYERS[opening_leader_idx]
        self._current_player = self._opening_leader
        self.playing_phase   = True

    def _find_declarer(self, team_parity: int, final_bid: int) -> str:
        """
        找出莊家：在叫牌者隊伍中，最早叫出與最終合約相同花色的玩家。
        """
        final_suit_ch = BIDDING_ACTIONS[final_bid][1]  # 'C'/'D'/'H'/'S'/'N'

        for i, action in enumerate(self._bid_history):
            if action <= 0 or action > 35:
                continue  # 跳過 Pass/X/XX
            bid_str = BIDDING_ACTIONS[action]
            if bid_str[1] != final_suit_ch:
                continue
            # 計算這步叫牌是哪位玩家
            bidder_idx = (self._dealer_idx + i) % 4
            if bidder_idx % 2 == team_parity:
                return PLAYERS[bidder_idx]

        # fallback: 使用最後叫牌者
        return PLAYERS[self._last_real_bidder_idx]

    # ------------------------------------------------------------------
    # 打牌階段
    # ------------------------------------------------------------------
    def get_legal_playing_card_indices(self) -> list:
        """回傳合法出牌的牌索引列表 (0~51)"""
        assert self.playing_phase, "目前是叫牌階段"
        mask = self._make_play_mask()
        return [i for i in range(52) if mask[i] == 1]

    def step_playing(self, card_idx: int):
        """
        執行一步出牌。
        card_idx: 0~51 的牌索引（直接使用，不需偏移）。
        回傳 (obs, reward, terminated, truncated, info)

        Reward 設計（站在莊家方角度）：
          - 每磴結束：+1.0（莊家方贏）/ -1.0（防守方贏）
          - 遊戲結束：根據完成/未完成合約計算額外 reward
        """
        assert self.playing_phase, "目前是叫牌階段"
        assert not self.game_over, "遊戲已結束"

        curr = self._current_player

        # 安全性檢查
        if card_idx not in self.hands[curr]:
            info = self._get_info()
            info['error'] = 'illegal_card'
            return self._get_obs(), -1.0, True, False, info

        # 出牌
        self.hands[curr].remove(card_idx)
        self._history_vec[card_idx] = 1.0
        self._current_trick_cards.append(card_idx)
        self._current_trick_players.append(PLAYERS.index(curr))

        reward = 0.0

        # 判斷磴結果
        if len(self._current_trick_cards) == 4:
            winner_idx = determine_trick_winner(
                self._current_trick_cards,
                self._current_trick_players,
                self._trump_idx,
            )
            winner_team = 'NS' if winner_idx in (0, 2) else 'EW'
            self.tricks_won[winner_team] += 1
            self._trick_count += 1

            # 每磴 reward（站在莊家方角度）
            declarer_team = 'NS' if self.declarer in ('N', 'S') else 'EW'
            reward = 1.0 if winner_team == declarer_team else -1.0

            self._current_trick_cards   = []
            self._current_trick_players = []
            self._current_player        = PLAYERS[winner_idx]

            # 遊戲結束
            if self._trick_count == 13:
                self.game_over     = True
                self.playing_phase = False
                reward += self._final_reward()
        else:
            # 同磴下一位
            next_idx = (PLAYERS.index(curr) + 1) % 4
            self._current_player = PLAYERS[next_idx]

        obs  = self._get_obs()
        info = self._get_info()
        return obs, reward, self.game_over, False, info

    def _final_reward(self) -> float:
        """遊戲結束後額外 reward（站在莊家方角度）"""
        level = int(self.contract[0])
        needed = level + 6
        declarer_team = 'NS' if self.declarer in ('N', 'S') else 'EW'
        d_tricks = self.tricks_won[declarer_team]

        if d_tricks >= needed:
            # 完成合約：+10 + 超磴獎勵
            return 10.0 + (d_tricks - needed) * 2.0
        else:
            # 未完成：-10 - 不足磴懲罰
            return -(10.0 + (needed - d_tricks) * 3.0)

    # ------------------------------------------------------------------
    # 相容 train_playing_rl.py 的 step（打牌階段通用入口，保留備用）
    # ------------------------------------------------------------------
    def step(self, action: int):
        if self.playing_phase:
            return self.step_playing(action)
        else:
            return self.step_bidding(action)