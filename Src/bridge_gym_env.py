"""
BridgeGymEnv - 橋牌 Gymnasium 環境 (規則引擎 + Gym 介面合一)

使用方式:
    from Src.bridge_gym_env import BridgeGymEnv
    env = BridgeGymEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

# ==========================================
# 常數定義
# ==========================================
SUITS = ['C', 'D', 'H', 'S']          # 梅花, 方塊, 紅心, 黑桃
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
PLAYERS = ['N', 'E', 'S', 'W']

# 叫牌動作表 (共 38 種)
# id=0: Pass, id=1: X(賭倍), id=2: XX(再賭倍), id=3~37: 1C~7NT
BIDDING_ACTIONS = ['Pass', 'X', 'XX']
for level in range(1, 8):
    for suit in ['C', 'D', 'H', 'S', 'NT']:
        BIDDING_ACTIONS.append(f"{level}{suit}")

# 標準化卡牌索引映射 (與 utils.py 一致: S→H→D→C)
_SUITS_STD = ['S', 'H', 'D', 'C']
CARD_TO_IDX = {}
IDX_TO_CARD = {}
for suit_i, suit in enumerate(_SUITS_STD):
    for rank_i, rank in enumerate(RANKS):
        std_idx = suit_i * 13 + rank_i
        card_name = f"{rank}{suit}"   # bridge_env 格式: {Rank}{Suit}
        CARD_TO_IDX[card_name] = std_idx
        IDX_TO_CARD[std_idx] = card_name

# 王牌映射
TRUMP_MAP = {'S': 0, 'H': 1, 'D': 2, 'C': 3, 'NT': 4}


class BridgeGymEnv(gym.Env):
    """
    橋牌 Gymnasium 環境 (叫牌 + 打牌 全階段)

    觀察空間 (Dict):
        phase: 0=叫牌, 1=打牌
        bidding_state: 78 維 (52 手牌 + 6 Extra + 20 歷史)
        playing_state: 165 維 (同原設定)

    動作空間:
        Discrete(90) — 結合叫牌與打牌
        0~37: 叫牌動作 (Pass, X, XX, 1C~7NT)
        38~89: 打牌動作 (卡牌索引 + 38)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Gymnasium 空間定義 (Dict 格式)
        self.observation_space = spaces.Dict({
            "phase": spaces.Discrete(2), # 0=叫牌, 1=打牌
            "bidding_state": spaces.Box(low=-10.0, high=40.0, shape=(78,), dtype=np.float32),
            "playing_state": spaces.Box(low=0.0, high=1.0, shape=(165,), dtype=np.float32)
        })
        
        # 動作空間: 0~37 叫牌, 38~89 打牌
        self.action_space = spaces.Discrete(90)

        # 建立牌組
        self.deck = [f"{r}{s}" for s in SUITS for r in RANKS]

        # 遊戲狀態 (由 reset 初始化)
        self.hands = {}
        self.bidding_history = []
        self.dealer_idx = 0
        self.current_player_idx = 0
        self.bidding_over = False
        self.contract = None
        self.declarer = None
        self.playing_phase = False
        self.current_trick = []
        self.led_suit = None
        self.tricks_won = {'NS': 0, 'EW': 0}
        self.game_over = False

        # Gym 額外追蹤
        self._full_history = []
        self._trick_count = 0
        self._dummy_player = None

    # ==========================================
    # 橋牌規則引擎 (原 bridge_env.py 的邏輯)
    # ==========================================

    def _sort_key(self, card):
        return (SUITS.index(card[1]), RANKS.index(card[0]))

    def _reset_game(self):
        """內部重置: 洗牌、發牌、清空狀態"""
        random.shuffle(self.deck)
        self.hands = {
            'N': sorted(self.deck[0:13],  key=self._sort_key),
            'E': sorted(self.deck[13:26], key=self._sort_key),
            'S': sorted(self.deck[26:39], key=self._sort_key),
            'W': sorted(self.deck[39:52], key=self._sort_key)
        }

        self.bidding_history = []
        self.dealer_idx = random.randint(0, 3)
        self.current_player_idx = self.dealer_idx
        self.bidding_over = False
        self.contract = None
        self.declarer = None

        self.playing_phase = False
        self.current_trick = []
        self.led_suit = None
        self.tricks_won = {'NS': 0, 'EW': 0}
        self.game_over = False

        self._full_history = []
        self._trick_count = 0
        self._dummy_player = None

    def get_current_player(self):
        return PLAYERS[self.current_player_idx]

    def get_legal_bidding_actions(self):
        """回傳合法叫牌動作列表 (包含 Pass, X, XX, 更高叫品)"""
        if self.bidding_over:
            return []

        legal_actions = [0]  # 永遠可以 Pass

        # 找出目前最高叫品
        highest_bid_idx = -1
        for bid_id in self.bidding_history:
            if bid_id >= 3:
                highest_bid_idx = bid_id

        # 加入更高的叫品
        if highest_bid_idx != -1:
            for bid_id in range(highest_bid_idx + 1, 38):
                legal_actions.append(bid_id)
        else:
            for bid_id in range(3, 38):
                legal_actions.append(bid_id)

        # --- X (賭倍) 判斷: 只能在對手叫品後使用 ---
        if highest_bid_idx >= 3:
            # 找最後一個非 Pass 的動作
            last_non_pass = None
            last_non_pass_player = None
            for i in range(len(self.bidding_history) - 1, -1, -1):
                if self.bidding_history[i] != 0:
                    last_non_pass = self.bidding_history[i]
                    last_non_pass_player = (self.dealer_idx + i) % 4
                    break

            if last_non_pass is not None and last_non_pass >= 3:
                # 最後的實質叫品必須是對手叫的
                current_team = self.current_player_idx % 2
                last_team = last_non_pass_player % 2
                if current_team != last_team:
                    legal_actions.append(1)  # X

        # --- XX (再賭倍) 判斷: 只能在對手 X 後使用 ---
        if len(self.bidding_history) > 0:
            last_non_pass = None
            last_non_pass_player = None
            for i in range(len(self.bidding_history) - 1, -1, -1):
                if self.bidding_history[i] != 0:
                    last_non_pass = self.bidding_history[i]
                    last_non_pass_player = (self.dealer_idx + i) % 4
                    break

            if last_non_pass == 1:  # 最後非 Pass 動作是 X
                current_team = self.current_player_idx % 2
                last_team = last_non_pass_player % 2
                if current_team != last_team:
                    legal_actions.append(2)  # XX

        return legal_actions

    def get_legal_playing_actions(self):
        """回傳合法出牌列表 (字串格式, 如 ['AS', 'KS'])"""
        if not self.playing_phase or self.game_over:
            return []

        current_hand = self.hands[self.get_current_player()]

        if len(self.current_trick) == 0:
            return list(current_hand)

        # 必須跟隨引牌花色
        led_suit = self.current_trick[0][1][1]
        following_cards = [c for c in current_hand if c[1] == led_suit]

        if following_cards:
            return following_cards
        else:
            return list(current_hand)

    def _step_bidding(self, action_id):
        """執行一步叫牌"""
        self.bidding_history.append(action_id)

        # 檢查叫牌是否結束 (最後三個都是 Pass，且至少叫了一輪)
        if len(self.bidding_history) >= 4 and all(act == 0 for act in self.bidding_history[-3:]):
            non_pass_bids = [(i, act) for i, act in enumerate(self.bidding_history) if act >= 3]
            if non_pass_bids:
                final_bid_idx, final_bid = non_pass_bids[-1]
                self.contract = BIDDING_ACTIONS[final_bid]

                # 莊家判斷: 主打方中最先叫出該花色的人
                contract_suit = self.contract[1:]
                winning_team = (self.dealer_idx + final_bid_idx) % 2

                for i, act in non_pass_bids:
                    if (self.dealer_idx + i) % 2 == winning_team:
                        if BIDDING_ACTIONS[act][1:] == contract_suit:
                            self.declarer = PLAYERS[(self.dealer_idx + i) % 4]
                            break

                self.bidding_over = True
                self.playing_phase = True
                # 首攻 = 莊家左手邊
                self.current_player_idx = (PLAYERS.index(self.declarer) + 1) % 4

                # 記錄夢家
                declarer_idx = PLAYERS.index(self.declarer)
                self._dummy_player = PLAYERS[(declarer_idx + 2) % 4]
            else:
                # 四家 Pass 流局
                self.contract = "Passed Out"
                self.bidding_over = True
                self.game_over = True
        else:
            self.current_player_idx = (self.current_player_idx + 1) % 4

    def _step_playing(self, card_to_play):
        """執行一步出牌"""
        current_player = self.get_current_player()
        self.hands[current_player].remove(card_to_play)
        self.current_trick.append((self.current_player_idx, card_to_play))
        self._full_history.append(card_to_play)

        if len(self.current_trick) == 4:
            self._evaluate_trick()
            if sum(self.tricks_won.values()) == 13:
                self.game_over = True
        else:
            self.current_player_idx = (self.current_player_idx + 1) % 4

    def _evaluate_trick(self):
        """判定本磴贏家"""
        trump_suit = self.contract[-1] if self.contract[-2:] != 'NT' else None
        led_suit = self.current_trick[0][1][1]

        highest_rank_val = -1
        winner_idx = -1

        for player_idx, card in self.current_trick:
            suit = card[1]
            rank_val = RANKS.index(card[0])

            if suit == trump_suit:
                effective_val = rank_val + 100
            elif suit == led_suit:
                effective_val = rank_val
            else:
                effective_val = -1

            if effective_val > highest_rank_val:
                highest_rank_val = effective_val
                winner_idx = player_idx

        winner = PLAYERS[winner_idx]
        if winner in ('N', 'S'):
            self.tricks_won['NS'] += 1
        else:
            self.tricks_won['EW'] += 1

        self.current_player_idx = winner_idx
        self.current_trick = []

    # ==========================================
    # Gymnasium 介面
    # ==========================================

    def reset(self, seed=None, options=None):
        """重置環境: 發牌 + 回傳初始觀察 (進入叫牌第一步)"""
        super().reset(seed=seed)
        self._reset_game()

        # 遊戲直接從叫牌階段開始，不自動叫牌了
        if self.render_mode == "human":
            print(f"新局開始，發牌者: {self.get_current_player()}")

        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        執行一步動作 (叫牌或出牌)。

        參數:
            action (int): 0~89
                - 0~37: 叫牌動作
                - 38~89: 打牌動作 (卡牌索引 + 38)
        """
        reward = 0.0
        penalty = 0.0
        
        # --- 叫牌階段 ---
        if not self.playing_phase:
            if action < 0 or action > 37:
                # 動作超出叫牌範圍，懲罰並隨機選一個合法動作
                penalty = -0.5
                legal_bids = self.get_legal_bidding_actions()
                action = random.choice(legal_bids) if legal_bids else 0
            else:
                legal_bids = self.get_legal_bidding_actions()
                if action not in legal_bids:
                    penalty = -0.5
                    action = random.choice(legal_bids) if legal_bids else 0
                    
            self._step_bidding(action)
            reward = penalty
            
            # 若流局，遊戲結束，給予些微懲罰或0
            if self.contract == "Passed Out":
                if self.render_mode == "human":
                    print("四家 Pass，流局！")
                return self._get_obs(), reward, self.game_over, False, self._get_info()
            
            # 剛結束叫牌時印出合約
            if self.playing_phase and self.render_mode == "human":
                print(f"合約確定: {self.contract} | 莊家: {self.declarer} | 夢家: {self._dummy_player}")
                
            return self._get_obs(), reward, self.game_over, False, self._get_info()

        # --- 打牌階段 ---
        else:
            card_idx = action - 38
            card_to_play = IDX_TO_CARD.get(card_idx)

            if card_idx < 0 or card_idx > 51 or card_to_play is None:
                penalty = -0.5
                legal_cards = self.get_legal_playing_actions()
                card_to_play = random.choice(legal_cards)
            else:
                legal_cards = self.get_legal_playing_actions()
                if card_to_play not in legal_cards:
                    card_to_play = random.choice(legal_cards)
                    penalty = -0.5

            # 記錄本磴開始前狀態
            trick_was_in_progress = len(self.current_trick) > 0
            ns_before = self.tricks_won['NS']
            ew_before = self.tricks_won['EW']

            # 執行出牌
            self._step_playing(card_to_play)

            # --- 計算獎勵 (站在莊家方角度) ---
            reward = penalty
            declarer_team = 'NS' if self.declarer in ('N', 'S') else 'EW'

            # 一磴結束時
            if len(self.current_trick) == 0 and (trick_was_in_progress or self.game_over):
                self._trick_count += 1
                declarer_won = self.tricks_won[declarer_team] > (
                    ns_before if declarer_team == 'NS' else ew_before
                )
                reward += 1.0 if declarer_won else -1.0

            # 遊戲結束時
            if self.game_over:
                contract_level = int(self.contract[0])
                needed_tricks = contract_level + 6
                declarer_tricks = self.tricks_won[declarer_team]

                if declarer_tricks >= needed_tricks:
                    overtricks = declarer_tricks - needed_tricks
                    reward += 10.0 + overtricks * 2.0
                else:
                    undertricks = needed_tricks - declarer_tricks
                    reward -= 10.0 + undertricks * 3.0

                if self.render_mode == "human":
                    result = "完成合約!" if declarer_tricks >= needed_tricks else "未完成"
                    print(f"遊戲結束! NS:{self.tricks_won['NS']} EW:{self.tricks_won['EW']} "
                          f"| 需要:{needed_tricks}磴 | {result}")

            return self._get_obs(), reward, self.game_over, False, self._get_info()

    def _get_obs(self):
        """將當前狀態轉為 Dict 觀察向量"""
        return {
            "phase": 1 if self.playing_phase else 0,
            "bidding_state": self._get_bidding_state(),
            "playing_state": self._get_playing_state()
        }

    def _get_bidding_state(self):
        """產生 78 維的叫牌特徵"""
        state = np.zeros(78, dtype=np.float32)
        if self.game_over:
            return state

        current_player = self.get_current_player()
        
        # [0:52] 手牌
        for card_str in self.hands[current_player]:
            idx = CARD_TO_IDX.get(card_str)
            if idx is not None:
                state[idx] = 1.0

        # [52:58] Extra 特徵 (目前使用前 4 維作為座位 N/E/S/W，後 2 維為 0)
        seat_idx = PLAYERS.index(current_player)
        state[52 + seat_idx] = 1.0

        # [58:78] 叫牌歷史 (最多前 20 個)
        # 用 -1 padding
        hist = self.bidding_history[-20:]
        hist_padded = [-1] * (20 - len(hist)) + hist
        for i, bid in enumerate(hist_padded):
            state[58 + i] = bid

        return state

    def _get_playing_state(self):
        """產生 165 維的打牌特徵"""
        state = np.zeros(165, dtype=np.float32)

        # [0:52] 已出牌歷史
        for card_str in self._full_history:
            idx = CARD_TO_IDX.get(card_str)
            if idx is not None:
                state[idx] = 1.0

        # [52:104] 當前玩家手牌
        if not self.game_over:
            current_player = self.get_current_player()
            for card_str in self.hands[current_player]:
                idx = CARD_TO_IDX.get(card_str)
                if idx is not None:
                    state[52 + idx] = 1.0

        # [104:156] 夢家手牌
        if self._dummy_player and not self.game_over:
            for card_str in self.hands[self._dummy_player]:
                idx = CARD_TO_IDX.get(card_str)
                if idx is not None:
                    state[104 + idx] = 1.0

        # [156:161] 王牌花色
        if self.contract and self.contract != "Passed Out":
            contract_suit = self.contract[-1] if self.contract[-2:] != 'NT' else 'NT'
            t_idx = TRUMP_MAP.get(contract_suit, 4)
            state[156 + t_idx] = 1.0

        # [161:165] 出牌順位
        if not self.game_over:
            pos = len(self.current_trick)
            if pos < 4:
                state[161 + pos] = 1.0

        return state

    def _get_info(self):
        """回傳額外資訊與 action_mask"""
        info = {
            'current_player': self.get_current_player() if not self.game_over else None,
            'contract': self.contract,
            'declarer': self.declarer,
            'dummy': self._dummy_player,
            'tricks_won': dict(self.tricks_won),
            'trick_count': self._trick_count,
        }

        # 建立 90 維的 action_mask
        action_mask = np.zeros(90, dtype=np.int8)

        if not self.game_over:
            if not self.playing_phase:
                # 叫牌遮罩 (0~37)
                legal_bids = self.get_legal_bidding_actions()
                for bid in legal_bids:
                    action_mask[bid] = 1
            else:
                # 打牌遮罩 (38~89)
                legal_cards = self.get_legal_playing_actions()
                for card in legal_cards:
                    idx = CARD_TO_IDX.get(card)
                    if idx is not None:
                        action_mask[38 + idx] = 1

        info['action_mask'] = action_mask
        return info

    def get_legal_actions(self):
        """回傳當前合法動作的索引列表 (0~89)"""
        if self.game_over:
            return []
            
        if not self.playing_phase:
            return self.get_legal_bidding_actions()
        else:
            legal_cards = self.get_legal_playing_actions()
            return [CARD_TO_IDX[card] + 38 for card in legal_cards if card in CARD_TO_IDX]

    def render(self):
        """人類可讀的遊戲狀態輸出"""
        if self.render_mode != "human":
            return

        current = self.get_current_player()
        hand = self.hands[current] if not self.game_over else []
        
        if not self.playing_phase:
            legal_bids = [BIDDING_ACTIONS[b] for b in self.get_legal_bidding_actions()] if not self.game_over else []
            history_names = [BIDDING_ACTIONS[b] for b in self.bidding_history]
            print(f"[{current}] 手牌: {hand} | 合法叫品: {legal_bids} | 歷史: {history_names}")
        else:
            legal = self.get_legal_playing_actions() if not self.game_over else []
            trick = [(PLAYERS[p], c) for p, c in self.current_trick]
            print(f"[{current}] 手牌: {hand} | 合法出牌: {legal} | 本磴: {trick} | "
                  f"比分 NS:{self.tricks_won['NS']} EW:{self.tricks_won['EW']}")


# ==========================================
# 測試: 用隨機動作跑一局
# ==========================================
if __name__ == "__main__":
    env = BridgeGymEnv(render_mode="human")
    obs, info = env.reset()

    print(f"\n觀察向量 (Dict):")
    for k, v in obs.items():
        print(f"  {k}: {type(v)} {np.shape(v) if isinstance(v, np.ndarray) else ''}")
    print(f"動作空間: {env.action_space}")
    print()

    total_reward = 0
    steps = 0

    while True:
        legal = env.get_legal_actions()
        action = random.choice(legal) if legal else env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        env.render()

        if terminated or truncated:
            break

    print(f"\n遊戲結束! 總步數: {steps} | 總獎勵: {total_reward:.1f}")
