"""
train_playing_rl.py — PPO 強化學習訓練出牌模型（最大化吃磴數）

目標：
  訓練模型讓指定隊伍（NS）極大化所取得的總磴數。

訓練模式（透過 --mode 參數切換）：
  pure         LegacyPolicyNet  隨機初始化 → PPO 訓練
  mix          LegacyPolicyNet  監督預訓練 policy_165dim_best.pth → PPO 微調
  resnet_pure  BridgePolicyNetResNet  隨機初始化 → PPO 訓練
  resnet_mix   BridgePolicyNetResNet  監督預訓練 policy_resnet_best.pth → PPO 微調

核心設計與修正：
  1. 叫牌階段：使用啟發式規則（Heuristic Bidding）取代純隨機叫牌，確保 RL 訓練環境具備合理的王牌邏輯。
  2. Reward 設計：
     - 步進獎勵：贏得單磴給予 +1.0，輸磴給予 -1.0。
     - 終端獎勵：最後一磴根據總吃磴數給予額外非線性成就獎勵，鼓勵建立壓倒性優勢。
  3. 對手池機制：EW 隊伍從歷史 Actor 快照池中隨機抽取，並使用 deepcopy 避免 GPU 記憶體洩漏。
"""

import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import random
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from Src.bridge_env import BridgeGymEnv, PLAYERS, BIDDING_ACTIONS
from Src.model import BridgePolicyNetResNet, LegacyPolicyNet, BridgePolicyNet


# ============================================================
# 超參數設定
# ============================================================
TOTAL_EPISODES           = 150000  # 總訓練局數
BATCH_GAMES              = 256     # 每次收集之局數（做為一次 PPO 更新的資料量）
PPO_EPOCHS               = 6       # 每次 PPO 更新重複疊代的 epoch 數
MINI_BATCH_SIZE          = 1024    # PPO mini-batch 大小
GAMMA                    = 0.99    # 獎勵折扣因子（Discount Factor）
GAE_LAMBDA               = 0.95    # GAE 參數 λ，控制偏差與變異數之權衡
CLIP_EPSILON             = 0.15    # PPO 裁切範圍（Clip Range），防止單次更新幅度過大
LR_ACTOR                 = 5e-5    # Actor 網路學習率
LR_CRITIC                = 1.5e-4  # Critic 網路學習率
ENTROPY_COEFF            = 0.01    # 熵正則化係數，數值越高越鼓勵模型探索
CRITIC_WARMUP            = 300     # 前 N 局僅訓練 Critic 網路以建立準確的價值基準
OPPONENT_UPDATE_INTERVAL = 2000    # 每 N 局將當前 Actor 之快照存入對手池
MAX_POOL_SIZE            = 20      # 對手池最大容量限制（FIFO 替換機制）
EVAL_INTERVAL            = 1000    # 評估與模型存檔之間隔局數
EVAL_GAMES               = 100     # 每次評估進行之總局數
SAVE_DIR                 = "Data/models"


# ============================================================
# 價值網路 (Critic Network)
# ============================================================
class PlayingValueNet(nn.Module):
    """Critic 模型：估計當前 165 維狀態之預期價值 V(s)"""
    def __init__(self, input_dim=165):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 移除最後一維，使輸出形狀與 Reward 陣列對齊 (Batch_Size,)
        return self.net(x).squeeze(-1)


# ============================================================
# 對手池 (Opponent Pool)
# ============================================================
class OpponentPool:
    """
    維護多個歷史 Actor 權重快照之對手池，供 EW 隊伍隨機抽取使用。
    以降低策略過擬合於單一對手之風險，實現非對稱自我對弈。
    """
    def __init__(self, max_size: int = MAX_POOL_SIZE):
        self._models: list = []
        self.max_size = max_size

    def add_snapshot(self, model: nn.Module) -> None:
        """將模型之深拷貝加入對手池，凍結梯度並強制進入 eval 模式以節省運算資源。"""
        snapshot = copy.deepcopy(model).cpu()
        snapshot.eval()
        for p in snapshot.parameters():
            p.requires_grad = False
        self._models.append(snapshot)
        if len(self._models) > self.max_size:
            self._models.pop(0)

    def sample(self, device: torch.device) -> nn.Module:
        """
        均勻隨機抽取一歷史模型。
        需使用 copy.deepcopy 避免原始存於 CPU 之快照被轉移至 GPU 導致記憶體洩漏。
        """
        model = random.choice(self._models)
        return copy.deepcopy(model).to(device)

    def __len__(self) -> int:
        return len(self._models)


# ============================================================
# 叫牌規則模組
# ============================================================
def random_bid(env):
    """自合法叫牌動作中隨機選取（已棄用，僅供測試保留）。"""
    legal = env.get_legal_bidding_actions()
    return random.choice(legal) if legal else 0

def heuristic_bid(env):
    """
    基於規則之啟發式叫牌器。
    根據南北軍 (NS) 手牌配置，自動決策合理之王牌合約。
    判定條件：尋找雙方配合總數達 8 張以上之花色作為王牌；若無，則定約為無王 (NT)。
    """
    n_hand = env.hands['N']
    s_hand = env.hands['S']
    # 確保兩者轉為 list 再相加，避免資料型態為 set 或 numpy array 導致錯誤
    ns_cards = list(n_hand) + list(s_hand)
    
    suit_counts = {'S': 0, 'H': 0, 'D': 0, 'C': 0}
    # 預設卡牌索引順序 (0~12=S, 13~25=H, 26~38=D, 39~51=C)
    suits_order = ['S', 'H', 'D', 'C'] 
    
    for card in ns_cards:
        # 若卡牌格式為整數 (0-51)，利用除以 13 判斷花色
        if isinstance(card, int) or isinstance(card, np.integer):
            suit = suits_order[card // 13]
        # 若卡牌格式為字串 (如 '2S')，抓取第二個字元
        elif isinstance(card, str) and len(card) >= 2:
            suit = card[1]
        else:
            continue
            
        if suit in suit_counts:
            suit_counts[suit] += 1
            
    best_suit = 'N'
    max_count = 0
    for suit, count in suit_counts.items():
        if count >= 8 and count > max_count:
            best_suit = suit
            max_count = count
            
    # 簡化設定：統一預設為 3 階合約
    target_bid_str = f"3{best_suit}" 
    
    try:
        action_id = BIDDING_ACTIONS.index(target_bid_str)
        return action_id
    except ValueError:
        return 0


# ============================================================
# 廣義優勢估計 (GAE)
# ============================================================
def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    """計算 Generalized Advantage Estimation (GAE)，平衡回報的偏差與變異數。"""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if (t == len(rewards) - 1 or dones[t]) else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (1 - float(dones[t])) * gae
        advantages.insert(0, gae)
    advantages = np.array(advantages, dtype=np.float32)
    returns    = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ============================================================
# 資料收集模組 (非對稱對弈)
# ============================================================
def collect_one_game_asymmetric(env, actor, opponent_pool, critic, device):
    """
    進行單局對弈並收集 PPO 訓練資料。
    NS 隊伍使用當前 Actor，EW 隊伍使用自對手池抽取之歷史快照。
    僅保留 NS 隊伍的軌跡資料 (Transitions) 以供梯度更新。
    """
    opponent = opponent_pool.sample(device)

    obs, info = env.reset()

    # 叫牌階段：由 NS 依據規則叫定合約，若逢 EW 則 Pass
    target_action = heuristic_bid(env)
    has_bid = False
    while not env.playing_phase and not env.game_over:
        curr_player = env.get_current_player()
        if curr_player in ('N', 'S') and not has_bid:
            obs, _, _, _, info = env.step_bidding(target_action)
            has_bid = True
        else:
            obs, _, _, _, info = env.step_bidding(0)

    if env.contract == "Passed Out":
        return [], {'passed_out': True, 'contract': 'Passed Out'}

    transitions = []
    trick_buffer = []

    while not env.game_over:
        acting_player = env.get_current_player()
        acting_team   = 'NS' if acting_player in ('N', 'S') else 'EW'
        state_vec     = obs["playing_state"]
        legal_indices = env.get_legal_playing_card_indices()

        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

        with torch.no_grad():
            net = actor if acting_team == 'NS' else opponent
            logits = net(state_tensor)

            mask = torch.full((52,), -1e9, device=device)
            for idx in legal_indices:
                mask[idx] = 0.0
            masked_logits = logits.squeeze(0) + mask

            dist     = torch.distributions.Categorical(logits=masked_logits)
            action   = dist.sample()
            log_prob = dist.log_prob(action)

            # Critic 僅需評估主訓練方 (NS) 之價值
            value = critic(state_tensor) if acting_team == 'NS' else torch.zeros(1, device=device)

        card_idx = action.item()
        tricks_before = dict(env.tricks_won)

        trick_buffer.append({
            'state':         state_vec.copy(),
            'action':        card_idx,
            'log_prob':      log_prob.item(),
            'value':         value.item(),
            'acting_player': acting_player,
            'acting_team':   acting_team,
            'legal_mask':    mask.cpu().numpy().copy(),
            'reward':        0.0,
            'done':          False,
        })

        obs, _, terminated, _, info = env.step_playing(card_idx)

        # 單磴結束判定
        if len(trick_buffer) == 4:
            for t in trick_buffer:
                t['done']   = env.game_over

                # 改為稀疏獎勵 (Sparse Reward)：僅在遊戲結束時，根據全局總吃磴數給予獎勵
                if env.game_over:
                    team_tricks = env.tricks_won[t['acting_team']]
                    # 基於 6.5 磴為及格線，計算最終分數
                    t['reward'] = (team_tricks - 6.5) / 6.5 * 10.0
                else:
                    t['reward'] = 0.0

                # 僅保留主訓練方 (NS) 之資料
                if t['acting_team'] == 'NS':
                    transitions.append(t)

            trick_buffer = []

    game_info = {
        'passed_out': False,
        'contract':   env.contract,
        'tricks_won': dict(env.tricks_won),
    }
    return transitions, game_info


# ============================================================
# PPO 策略更新模組
# ============================================================
def ppo_update(actor, critic, optimizer_actor, optimizer_critic,
               all_games, device, update_actor=True):
    """
    執行 PPO 之梯度更新。
    update_actor: 設為 False 時僅更新 Critic，適用於訓練初期之價值基準建立。
    """
    processed = []
    for game in all_games:
        player_data = {}
        for t in game:
            p = t['acting_player']
            player_data.setdefault(p, []).append(t)

        for p, steps in player_data.items():
            rewards   = [s['reward'] for s in steps]
            values    = [s['value']  for s in steps]
            dones     = [False] * len(steps)
            dones[-1] = True

            advantages, returns = compute_gae(rewards, values, dones)
            for i, s in enumerate(steps):
                s['advantage'] = advantages[i]
                s['return']    = returns[i]
                processed.append(s)

    if not processed:
        return 0.0, 0.0

    states        = torch.FloatTensor(np.array([t['state']      for t in processed])).to(device)
    actions       = torch.LongTensor([t['action']               for t in processed]).to(device)
    old_log_probs = torch.FloatTensor([t['log_prob']            for t in processed]).to(device)
    advantages_t  = torch.FloatTensor([t['advantage']           for t in processed]).to(device)
    returns_t     = torch.FloatTensor([t['return']              for t in processed]).to(device)
    legal_masks   = torch.FloatTensor(np.array([t['legal_mask'] for t in processed])).to(device)

    # Advantage 正規化，提升訓練穩定性
    if len(advantages_t) > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    total_actor_loss  = 0.0
    total_critic_loss = 0.0
    n_updates = 0

    for _ in range(PPO_EPOCHS):
        indices = torch.randperm(len(processed))
        for start in range(0, len(processed), MINI_BATCH_SIZE):
            end    = min(start + MINI_BATCH_SIZE, len(processed))
            mb_idx = indices[start:end]

            mb_states  = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_lp  = old_log_probs[mb_idx]
            mb_adv     = advantages_t[mb_idx]
            mb_ret     = returns_t[mb_idx]
            mb_masks   = legal_masks[mb_idx]

            # 更新 Critic
            critic.train()
            values_pred = critic(mb_states)
            critic_loss = F.mse_loss(values_pred, mb_ret)

            optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer_critic.step()

            # 更新 Actor (視階段決定是否執行)
            if update_actor:
                actor.train()
                # 強制將 BatchNorm 固定在 eval 模式，避免小批次破壞 running stats
                for m in actor.modules():
                    if isinstance(m, nn.BatchNorm1d):
                        m.eval()
                        
                logits        = actor(mb_states)
                masked_logits = logits + mb_masks
                dist          = torch.distributions.Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy       = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEFF * entropy

                optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                optimizer_actor.step()

                total_actor_loss += actor_loss.item()

            total_critic_loss += critic_loss.item()
            n_updates += 1

    actor.eval()
    critic.eval()
    return (total_actor_loss  / max(n_updates, 1),
            total_critic_loss / max(n_updates, 1))


# ============================================================
# 模型評估模組
# ============================================================
def evaluate(actor, device, n_games=EVAL_GAMES, baseline=None):
    """
    於固定局數下評估 Actor 模型之表現。
    baseline: 若提供則做為對手 (EW) 模型，否則 EW 使用隨機策略。
    """
    env = BridgeGymEnv()
    total_ns_tricks = 0
    valid_games = 0

    actor.eval()
    if baseline is not None:
        baseline.eval()

    for _ in range(n_games):
        obs, info = env.reset()

        # 評估階段同採規則叫牌
        target_action = heuristic_bid(env)
        has_bid = False
        while not env.playing_phase and not env.game_over:
            curr_player = env.get_current_player()
            if curr_player in ('N', 'S') and not has_bid:
                obs, _, _, _, info = env.step_bidding(target_action)
                has_bid = True
            else:
                obs, _, _, _, info = env.step_bidding(0)

        if env.contract == "Passed Out":
            continue

        valid_games += 1

        while not env.game_over:
            state_vec     = obs["playing_state"]
            legal_indices = env.get_legal_playing_card_indices()
            curr_player   = env.get_current_player()
            curr_team     = 'NS' if curr_player in ('N', 'S') else 'EW'

            if curr_team == 'NS':
                state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = actor(state_tensor)
                    mask   = torch.full((52,), -1e9, device=device)
                    for idx in legal_indices:
                        mask[idx] = 0.0
                    logits = logits.squeeze(0) + mask
                    action = logits.argmax().item()
            else:
                if baseline is not None:
                    state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = baseline(state_tensor)
                        mask   = torch.full((52,), -1e9, device=device)
                        for idx in legal_indices:
                            mask[idx] = 0.0
                        logits = logits.squeeze(0) + mask
                        action = logits.argmax().item()
                else:
                    action = random.choice(legal_indices)

            obs, _, _, _, info = env.step_playing(action)

        total_ns_tricks += env.tricks_won['NS']

    avg_ns_tricks = total_ns_tricks / max(valid_games, 1)
    return avg_ns_tricks, valid_games


# ============================================================
# 訓練主程序
# ============================================================
def main(mode: str = "mix"):
    """主訓練迴圈控制函數"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------
    # 根據 mode 決定模型架構、checkpoint 路徑與監督初始化來源
    # -------------------------------------------------------
    if mode == "pure":
        mode_label      = "純深度強化學習 (Pure RL — Legacy MLP)"
        ActorClass      = LegacyPolicyNet
        actor_kwargs    = {"input_dim": 165}
        supervised_init = os.path.join(SAVE_DIR, "policy_165dim_best.pth")
        ckpt_best       = "playing_rl_pure_best.pth"
        ckpt_latest     = "playing_rl_pure_latest.pth"
        ckpt_critic     = "playing_rl_pure_critic.pth"
        is_legacy       = True
        is_mix          = False

    elif mode == "mix":
        mode_label      = "監督預訓練 + 強化學習微調 (Mix RL — Legacy MLP)"
        ActorClass      = LegacyPolicyNet
        actor_kwargs    = {"input_dim": 165}
        supervised_init = os.path.join(SAVE_DIR, "policy_165dim_best.pth")
        ckpt_best       = "playing_rl_mix_best.pth"
        ckpt_latest     = "playing_rl_mix_latest.pth"
        ckpt_critic     = "playing_rl_mix_critic.pth"
        is_legacy       = True
        is_mix          = True

    elif mode == "resnet_pure":
        mode_label      = "純深度強化學習 (Pure RL — ResNet)"
        ActorClass      = BridgePolicyNetResNet
        actor_kwargs    = {"input_dim": 165, "hidden_dim": 512, "output_dim": 52, "num_blocks": 4}
        supervised_init = os.path.join(SAVE_DIR, "policy_resnet_best.pth")
        ckpt_best       = "playing_rl_resnet_pure_best.pth"
        ckpt_latest     = "playing_rl_resnet_pure_latest.pth"
        ckpt_critic     = "playing_rl_resnet_pure_critic.pth"
        is_legacy       = False
        is_mix          = False

    else:  # resnet_mix
        mode_label      = "監督預訓練 + 強化學習微調 (Mix RL — ResNet)"
        ActorClass      = BridgePolicyNetResNet
        actor_kwargs    = {"input_dim": 165, "hidden_dim": 512, "output_dim": 52, "num_blocks": 4}
        supervised_init = os.path.join(SAVE_DIR, "policy_resnet_best.pth")
        ckpt_best       = "playing_rl_resnet_mix_best.pth"
        ckpt_latest     = "playing_rl_resnet_mix_latest.pth"
        ckpt_critic     = "playing_rl_resnet_mix_critic.pth"
        is_legacy       = False
        is_mix          = True

    print(f"=== 橋牌出牌策略強化學習 | 模式: {mode_label} | 運算設備: {device} ===")
    print(f"    Actor 架構: {ActorClass.__name__}")
    print("模型超參數設定:")
    print(f"  BATCH_GAMES={BATCH_GAMES}, PPO_EPOCHS={PPO_EPOCHS}, MINI_BATCH_SIZE={MINI_BATCH_SIZE}")
    print(f"  LR_ACTOR={LR_ACTOR}, LR_CRITIC={LR_CRITIC}, ENTROPY_COEFF={ENTROPY_COEFF}\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 初始化 Actor 網路
    actor   = ActorClass(**actor_kwargs).to(device)
    rl_ckpt = os.path.join(SAVE_DIR, ckpt_latest)

    # Legacy 模式：將 Dropout 替換為 Identity，避免 PPO ratio 計算出現隨機性
    if is_legacy:
        actor.fc_net[3] = nn.Identity()
        actor.fc_net[6] = nn.Identity()

    if os.path.exists(rl_ckpt):
        actor.load_state_dict(torch.load(rl_ckpt, map_location=device))
        print(f"[Actor 載入] 恢復 RL 訓練進度: {rl_ckpt}")
    elif is_mix and os.path.exists(supervised_init):
        actor.load_state_dict(torch.load(supervised_init, map_location=device))
        print(f"[Actor 載入] 初始化為監督學習預訓練模型: {supervised_init}")
    else:
        print("[Actor 載入] 未偵測到模型權重，採用隨機初始化。")
    actor.eval()

    # 初始化對手池與評估基準（與 Actor 使用相同架構）
    pool          = OpponentPool(max_size=MAX_POOL_SIZE)
    eval_baseline = None

    if os.path.exists(supervised_init):
        supervised_model = ActorClass(**actor_kwargs)
        supervised_model.load_state_dict(torch.load(supervised_init, map_location="cpu"))
        pool.add_snapshot(supervised_model)

        eval_baseline = ActorClass(**actor_kwargs).to(device)
        eval_baseline.load_state_dict(torch.load(supervised_init, map_location=device))
        eval_baseline.eval()
        for p in eval_baseline.parameters():
            p.requires_grad = False
        print("[對手設定] 監督學習預訓練模型已加入對手池，並設定為評估基準對手。")
    else:
        print("[對手設定] 無監督預訓練模型，評估時將採用隨機出牌之對手。")

    pool.add_snapshot(actor)

    # 初始化 Critic 網路
    critic = PlayingValueNet(input_dim=165).to(device)
    critic_ckpt = os.path.join(SAVE_DIR, ckpt_critic)
    if os.path.exists(critic_ckpt):
        critic.load_state_dict(torch.load(critic_ckpt, map_location=device))
        print(f"[Critic 載入] 恢復預測網路權重: {critic_ckpt}\n")
    else:
        print("[Critic 載入] 採用隨機初始化。\n")
    critic.eval()

    # 優化器配置
    optimizer_actor  = optim.Adam(actor.parameters(),  lr=LR_ACTOR,  eps=1e-5)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC, eps=1e-5)

    # 學習率排程：根據實際的更新次數配置
    total_updates = TOTAL_EPISODES // BATCH_GAMES
    scheduler_actor  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_actor,  T_max=total_updates, eta_min=1e-6)
    scheduler_critic = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_critic, T_max=total_updates, eta_min=1e-6)

    env = BridgeGymEnv()
    best_tricks  = 0.0
    episode      = 0
    opponent_ver = 0 

    pbar = tqdm(total=TOTAL_EPISODES, desc="訓練進度 (PPO)")
    while episode < TOTAL_EPISODES:

        # 定期更新對手池
        if episode > 0 and episode % OPPONENT_UPDATE_INTERVAL < BATCH_GAMES:
            pool.add_snapshot(actor)
            opponent_ver += 1
            tqdm.write(f"  [環境更新] 第 {episode} 局：新策略快照已存入對手池 (版本: v{opponent_ver})")

        all_games = []
        batch_ns_tricks = []
        batch_ew_tricks = []

        for _ in range(BATCH_GAMES):
            transitions, game_info = collect_one_game_asymmetric(
                env, actor, pool, critic, device
            )
            if transitions:
                all_games.append(transitions)
            if not game_info.get('passed_out', False):
                tw = game_info.get('tricks_won', {})
                batch_ns_tricks.append(tw.get('NS', 0))
                batch_ew_tricks.append(tw.get('EW', 0))
            episode += 1

        in_warmup = episode <= CRITIC_WARMUP

        # 執行 PPO 梯度更新
        if all_games:
            actor_loss, critic_loss = ppo_update(
                actor, critic, optimizer_actor, optimizer_critic,
                all_games, device,
                update_actor=not in_warmup,
            )
            # 只有在確實有執行梯度更新時，才推進 Learning Rate
            scheduler_critic.step()
            if not in_warmup:
                scheduler_actor.step()
        else:
            actor_loss, critic_loss = 0.0, 0.0

        avg_ns = np.mean(batch_ns_tricks) if batch_ns_tricks else 0.0
        avg_ew = np.mean(batch_ew_tricks) if batch_ew_tricks else 0.0
        pbar.update(BATCH_GAMES)
        pbar.set_postfix(
            NS_Tricks = f"{avg_ns:.1f}",
            EW_Tricks = f"{avg_ew:.1f}",
            Act_Loss  = f"{actor_loss:.4f}",
            Crt_Loss  = f"{critic_loss:.4f}"
        )

        # 模型評估與保存
        if episode % EVAL_INTERVAL < BATCH_GAMES:
            avg_tricks, valid = evaluate(actor, device, baseline=eval_baseline)
            baseline_name = "監督模型" if eval_baseline is not None else "隨機"
            
            tqdm.write(f"\n[階段性評估] 累積局數: {episode} | 測試局數: {valid}/{EVAL_GAMES}")
            tqdm.write(f"對手陣營: {baseline_name} (池版本: v{opponent_ver})")
            tqdm.write(f"NS 陣營平均取得磴數: {avg_tricks:.2f} / 13.00")

            if avg_tricks > best_tricks:
                best_tricks = avg_tricks
                torch.save(actor.state_dict(), os.path.join(SAVE_DIR, ckpt_best))
                tqdm.write(f"  -> 已儲存當前最佳權重 (歷史最佳磴數: {best_tricks:.2f})")

            torch.save(actor.state_dict(), os.path.join(SAVE_DIR, ckpt_latest))
            torch.save(critic.state_dict(), os.path.join(SAVE_DIR, ckpt_critic))

    pbar.close()
    print(f"\n=== 訓練流程結束 | 歷史最佳 NS 平均磴數: {best_tricks:.2f} / 13.00 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="橋牌出牌策略強化學習執行腳本",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["pure", "mix", "resnet_pure", "resnet_mix"],
        default="mix",
        help=(
            "pure        : LegacyPolicyNet，隨機初始化 → PPO 訓練\n"
            "mix         : LegacyPolicyNet，policy_165dim_best.pth → PPO 微調 (預設)\n"
            "resnet_pure : BridgePolicyNetResNet，隨機初始化 → PPO 訓練\n"
            "resnet_mix  : BridgePolicyNetResNet，policy_resnet_best.pth → PPO 微調"
        )
    )
    args = parser.parse_args()
    main(mode=args.mode)