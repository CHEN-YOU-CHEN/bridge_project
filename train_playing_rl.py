"""
train_playing_rl.py — PPO 強化學習訓練出牌模型

流程:
  1. 叫牌階段: 使用凍結的 BidAgent (bidding_model2.pth)
  2. 打牌階段: 使用訓練中的 BridgePolicyNet (Actor)
  3. 每磴 + 遊戲結束時計算 reward
  4. PPO 更新 Actor + Critic

參考 train.py 的模型架構與風格
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from Src.bridge_gym_env import BridgeGymEnv, BIDDING_ACTIONS, CARD_TO_IDX, IDX_TO_CARD, PLAYERS
from Src.model import BridgePolicyNet
from Src.bid_agent import BidAgent

# ==========================================
# 超參數
# ==========================================
TOTAL_EPISODES   = 50000   # 總自我對弈局數
BATCH_GAMES      = 64      # 每次收集幾局後做 PPO 更新
PPO_EPOCHS       = 4       # 每次更新重複幾個 epoch
MINI_BATCH_SIZE  = 512     # PPO mini-batch 大小
GAMMA            = 0.99    # 折扣因子
GAE_LAMBDA       = 0.95    # GAE λ
CLIP_EPSILON     = 0.2     # PPO clip 範圍
LR_ACTOR         = 1e-5    # Actor 學習率 (微調，比 train.py 更小)
LR_CRITIC        = 5e-5    # Critic 學習率
ENTROPY_COEFF    = 0.01    # 熵正則化係數
EVAL_INTERVAL    = 500     # 每幾局做一次評估
EVAL_GAMES       = 50      # 評估時打幾局
SAVE_DIR         = "Data/models"

# ==========================================
# Critic 網路 (估計出牌狀態價值)
# ==========================================
class PlayingValueNet(nn.Module):
    """Critic: 估計 165 維出牌觀察的狀態價值 V(s)"""
    def __init__(self, input_dim=165):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ==========================================
# GAE 計算
# ==========================================
def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    """計算 Generalized Advantage Estimation"""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1] if not dones[t] else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    advantages = np.array(advantages, dtype=np.float32)
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ==========================================
# Reward 設計 (站在當前出牌者的隊伍角度)
# ==========================================
def compute_player_reward(env_reward, acting_player, declarer):
    """
    將環境的 reward (莊家角度) 轉換為出牌者角度。
    莊家方玩家：直接使用 env_reward
    防守方玩家：反轉 env_reward

    Reward 來源 (環境已計算):
    - 贏一磴: +1.0 / 輸一磴: -1.0
    - 合約完成: +10 + 超磴×2
    - 合約失敗: -(10 + 不足磴×3)
    """
    if declarer is None:
        return 0.0
    declarer_team = 'NS' if declarer in ('N', 'S') else 'EW'
    player_team = 'NS' if acting_player in ('N', 'S') else 'EW'
    if player_team == declarer_team:
        return env_reward
    else:
        return -env_reward


# ==========================================
# 收集一局出牌資料
# ==========================================
def collect_one_game(env, actor, critic, bid_agent, device):
    """
    執行一局完整遊戲，收集打牌階段的 trajectory。
    叫牌階段使用凍結的 BidAgent。

    回傳: list of dicts (打牌 transitions), game_info dict
    """
    obs, info = env.reset()

    # ----- 叫牌階段: 使用凍結的 BidAgent -----
    while not env.playing_phase and not env.game_over:
        legal_actions = env.get_legal_bidding_actions()
        action, _ = bid_agent.select_action(obs["bidding_state"], legal_actions)
        obs, _, _, _, info = env.step_bidding(action)

    if env.contract == "Passed Out":
        return [], {'passed_out': True, 'contract': 'Passed Out'}

    declarer = env.declarer

    # ----- 打牌階段: 收集訓練資料 -----
    transitions = []

    while not env.game_over:
        acting_player = env.get_current_player()
        state_vec = obs["playing_state"]
        legal_indices = env.get_legal_playing_card_indices()

        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = actor(state_tensor)
            # 合法動作遮罩
            mask = torch.full((52,), float('-inf'), device=device)
            for idx in legal_indices:
                mask[idx] = 0.0
            masked_logits = logits.squeeze(0) + mask
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            value = critic(state_tensor)

        card_idx = action.item()

        transitions.append({
            'state': state_vec.copy(),
            'action': card_idx,
            'log_prob': log_prob.item(),
            'value': value.item(),
            'acting_player': acting_player,
            'legal_mask': mask.cpu().numpy().copy(),
            'reward': 0.0,   # 先設 0，step 後再填
            'done': False,
        })

        # 環境 step
        obs, env_reward, terminated, truncated, info = env.step_playing(card_idx)

        # 計算此步的 player reward
        player_reward = compute_player_reward(env_reward, acting_player, declarer)
        transitions[-1]['reward'] = player_reward
        transitions[-1]['done'] = env.game_over

    game_info = {
        'passed_out': False,
        'contract': env.contract,
        'declarer': declarer,
        'tricks_won': dict(env.tricks_won),
    }
    return transitions, game_info


# ==========================================
# PPO 更新
# ==========================================
def ppo_update(actor, critic, optimizer_actor, optimizer_critic,
               all_transitions, device):
    """對收集到的所有 transitions 做 PPO 更新"""

    # --- 按玩家分組計算 GAE ---
    game_transitions = []
    current_game = []
    for t in all_transitions:
        current_game.append(t)
        if t['done']:
            game_transitions.append(current_game)
            current_game = []

    processed = []
    for game in game_transitions:
        player_data = {}
        for t in game:
            p = t['acting_player']
            if p not in player_data:
                player_data[p] = []
            player_data[p].append(t)

        for p, steps in player_data.items():
            rewards = [s['reward'] for s in steps]
            values = [s['value'] for s in steps]
            dones = [False] * len(steps)
            dones[-1] = True
            advantages, returns = compute_gae(rewards, values, dones)
            for i, s in enumerate(steps):
                s['advantage'] = advantages[i]
                s['return'] = returns[i]
                processed.append(s)

    if not processed:
        return 0.0, 0.0

    # --- 轉為 tensor ---
    states = torch.FloatTensor(np.array([t['state'] for t in processed])).to(device)
    actions = torch.LongTensor([t['action'] for t in processed]).to(device)
    old_log_probs = torch.FloatTensor([t['log_prob'] for t in processed]).to(device)
    advantages_t = torch.FloatTensor([t['advantage'] for t in processed]).to(device)
    returns_t = torch.FloatTensor([t['return'] for t in processed]).to(device)
    legal_masks = torch.FloatTensor(np.array([t['legal_mask'] for t in processed])).to(device)

    # 正規化 advantage
    if len(advantages_t) > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    # --- PPO epoch ---
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    n_updates = 0

    for _ in range(PPO_EPOCHS):
        indices = torch.randperm(len(processed))
        for start in range(0, len(processed), MINI_BATCH_SIZE):
            end = min(start + MINI_BATCH_SIZE, len(processed))
            mb_idx = indices[start:end]

            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_lp = old_log_probs[mb_idx]
            mb_adv = advantages_t[mb_idx]
            mb_ret = returns_t[mb_idx]
            mb_masks = legal_masks[mb_idx]

            # Actor forward
            actor.train()
            logits = actor(mb_states)
            masked_logits = logits + mb_masks
            dist = torch.distributions.Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            # PPO clipped loss
            ratio = torch.exp(new_log_probs - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * mb_adv
            actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEFF * entropy

            optimizer_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer_actor.step()

            # Critic forward
            critic.train()
            values = critic(mb_states)
            critic_loss = F.mse_loss(values, mb_ret)

            optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer_critic.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            n_updates += 1

    actor.eval()
    critic.eval()
    avg_actor = total_actor_loss / max(n_updates, 1)
    avg_critic = total_critic_loss / max(n_updates, 1)
    return avg_actor, avg_critic


# ==========================================
# 評估
# ==========================================
def evaluate(actor, bid_agent, device, n_games=EVAL_GAMES):
    """用目前的 actor 打 n_games 局，回傳平均表現"""
    env = BridgeGymEnv()
    total_reward = 0.0
    contracts_made = 0
    total_tricks = 0
    valid_games = 0

    for _ in range(n_games):
        obs, info = env.reset()

        # 叫牌 (凍結)
        while not env.playing_phase and not env.game_over:
            legal_actions = env.get_legal_bidding_actions()
            action, _ = bid_agent.select_action(obs["bidding_state"], legal_actions)
            obs, _, _, _, info = env.step_bidding(action)

        if env.contract == "Passed Out":
            continue

        declarer = env.declarer
        valid_games += 1

        # 打牌 (用 actor greedy)
        while not env.game_over:
            state_vec = obs["playing_state"]
            legal_indices = env.get_legal_playing_card_indices()
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = actor(state_tensor)
                mask = torch.full((52,), float('-inf'), device=device)
                for idx in legal_indices:
                    mask[idx] = 0.0
                logits = logits.squeeze(0) + mask
                action = logits.argmax().item()
            obs, _, _, _, info = env.step_playing(action)

        level = int(env.contract[0])
        needed = level + 6
        d_team = 'NS' if declarer in ('N', 'S') else 'EW'
        d_tricks = env.tricks_won[d_team]
        total_tricks += d_tricks

        if d_tricks >= needed:
            contracts_made += 1
            overtricks = d_tricks - needed
            total_reward += 10.0 + overtricks * 2.0
        else:
            undertricks = needed - d_tricks
            total_reward -= 10.0 + undertricks * 3.0

    avg_reward = total_reward / max(valid_games, 1)
    made_rate = contracts_made / max(valid_games, 1)
    avg_tricks = total_tricks / max(valid_games, 1)
    return avg_reward, made_rate, avg_tricks, valid_games


# ==========================================
# 主訓練迴圈
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 出牌 RL 訓練 (PPO) | 設備: {device} ===\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 初始化 Actor (從預訓練權重開始, 參考 train.py) ---
    actor = BridgePolicyNet(input_dim=165, hidden_dim=1024, output_dim=52).to(device)
    play_model_path = os.path.join(SAVE_DIR, "policy_165dim_latest.pth")
    if os.path.exists(play_model_path):
        actor.load_state_dict(torch.load(play_model_path, map_location=device))
        print(f"已載入預訓練出牌模型: {play_model_path}")
    else:
        print("警告: 找不到預訓練出牌模型，從隨機權重開始訓練")
    actor.eval()

    # --- 初始化 Critic ---
    critic = PlayingValueNet().to(device)
    critic.eval()

    # --- 凍結的 BidAgent ---
    bid_agent = BidAgent(
        model_path=os.path.join(SAVE_DIR, "bidding_model2.pth"),
        device=device
    )

    # --- 優化器 ---
    optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # --- 訓練迴圈 ---
    env = BridgeGymEnv()
    best_reward = float('-inf')
    episode = 0

    pbar = tqdm(total=TOTAL_EPISODES, desc="Playing RL")
    while episode < TOTAL_EPISODES:
        # 收集 BATCH_GAMES 局的資料
        all_transitions = []
        batch_rewards = []

        for _ in range(BATCH_GAMES):
            transitions, game_info = collect_one_game(
                env, actor, critic, bid_agent, device
            )
            all_transitions.extend(transitions)
            if transitions:
                game_reward = sum(t['reward'] for t in transitions) / len(transitions)
                batch_rewards.append(game_reward)
            episode += 1

        # PPO 更新
        if all_transitions:
            actor_loss, critic_loss = ppo_update(
                actor, critic, optimizer_actor, optimizer_critic,
                all_transitions, device
            )
        else:
            actor_loss, critic_loss = 0.0, 0.0

        avg_batch_reward = np.mean(batch_rewards) if batch_rewards else 0.0
        pbar.update(BATCH_GAMES)
        pbar.set_postfix(
            reward=f"{avg_batch_reward:.2f}",
            a_loss=f"{actor_loss:.4f}",
            c_loss=f"{critic_loss:.4f}"
        )

        # 定期評估與儲存
        if episode % EVAL_INTERVAL < BATCH_GAMES:
            avg_reward, made_rate, avg_tricks, valid = evaluate(
                actor, bid_agent, device
            )
            print(f"\n[Eval @ {episode}] Avg Reward: {avg_reward:.2f} | "
                  f"合約完成率: {made_rate:.1%} | 平均磴數: {avg_tricks:.1f} | "
                  f"有效局數: {valid}/{EVAL_GAMES}")

            # 儲存最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(actor.state_dict(),
                           os.path.join(SAVE_DIR, "playing_rl_best.pth"))
                print(f"  → 已儲存最佳模型 (reward: {best_reward:.2f})")

            # 總是儲存最新
            torch.save(actor.state_dict(),
                       os.path.join(SAVE_DIR, "playing_rl_latest.pth"))
            torch.save(critic.state_dict(),
                       os.path.join(SAVE_DIR, "playing_rl_critic.pth"))

    pbar.close()
    print(f"\n=== 訓練完成! 最佳 reward: {best_reward:.2f} ===")


if __name__ == "__main__":
    main()
