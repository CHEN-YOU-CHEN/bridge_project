"""
train_bidding_rl.py — PPO 強化學習訓練叫牌模型

流程:
  1. 叫牌階段: 使用訓練中的 BiddingModel (Actor)
  2. 打牌階段: 使用凍結的 PlayAgent (policy_165dim_latest.pth)
  3. 遊戲結束後根據結果計算叫牌 reward
  4. PPO 更新 Actor + Critic
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
from policy import BiddingModel
from Src.bridge_gym_env import BridgeGymEnv, BIDDING_ACTIONS, CARD_TO_IDX, IDX_TO_CARD, PLAYERS
from Src.model import BridgePolicyNet
from Src.play_agent import PlayAgent

# ==========================================
# 超參數
# ==========================================
TOTAL_EPISODES   = 50000   # 總自我對弈局數
BATCH_GAMES      = 64      # 每次收集幾局後做 PPO 更新
PPO_EPOCHS       = 4       # 每次更新重複幾個 epoch
MINI_BATCH_SIZE  = 256     # PPO mini-batch 大小
GAMMA            = 0.99    # 折扣因子
GAE_LAMBDA       = 0.95    # GAE λ
CLIP_EPSILON     = 0.2     # PPO clip 範圍
LR_ACTOR         = 3e-5    # Actor 學習率 (微調)
LR_CRITIC        = 1e-4    # Critic 學習率
ENTROPY_COEFF    = 0.02    # 熵正則化係數
EVAL_INTERVAL    = 500     # 每幾局做一次評估
EVAL_GAMES       = 50      # 評估時打幾局
SAVE_DIR         = "Data/models"

# ==========================================
# Critic 網路 (估計叫牌狀態價值)
# ==========================================
class BiddingValueNet(nn.Module):
    """Critic: 估計 78 維叫牌觀察的狀態價值 V(s)"""
    def __init__(self, input_dim=78):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
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
    # 反向遍歷
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
# 收集一局叫牌資料
# ==========================================
def collect_one_game(env, actor, critic, play_agent, device):
    """
    執行一局完整遊戲，收集叫牌階段的 trajectory。
    打牌階段使用凍結的 PlayAgent。

    回傳: list of dicts, 每個 dict 包含一步叫牌的 transition
    """
    obs, info = env.reset()
    transitions = []

    # ----- 叫牌階段: 收集訓練資料 -----
    while not env.playing_phase and not env.game_over:
        acting_player = env.get_current_player()
        state_vec = obs["bidding_state"]
        legal_actions = env.get_legal_bidding_actions()

        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = actor(state_tensor)
            # 合法動作遮罩
            mask = torch.full((38,), float('-inf'), device=device)
            for a in legal_actions:
                mask[a] = 0.0
            masked_logits = (logits.squeeze(0) + mask)
            dist = torch.distributions.Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            value = critic(state_tensor)

        # 記錄 transition
        transitions.append({
            'state': state_vec.copy(),
            'action': action.item(),
            'log_prob': log_prob.item(),
            'value': value.item(),
            'acting_player': acting_player,
            'legal_mask': mask.cpu().numpy().copy(),
            'reward': 0.0,   # 叫牌期間 reward 先設 0，結束後回填
            'done': False,
        })

        obs, reward, terminated, truncated, info = env.step_bidding(action.item())

    # ----- 檢查流局 -----
    if env.contract == "Passed Out":
        # 流局 → 所有叫牌步驟都給小懲罰
        for t in transitions:
            t['reward'] = -1.0
            t['done'] = True
        return transitions

    # ----- 打牌階段: 使用凍結 PlayAgent -----
    while not env.game_over:
        legal_indices = env.get_legal_playing_card_indices()
        card_idx, card_name, confidence = play_agent.select_action(
            obs["playing_state"], legal_indices
        )
        obs, reward, terminated, truncated, info = env.step_playing(card_idx)

    # ----- 計算叫牌 reward (根據遊戲結果) -----
    contract_level = int(env.contract[0])
    needed_tricks = contract_level + 6
    declarer_team = 'NS' if env.declarer in ('N', 'S') else 'EW'
    declarer_tricks = env.tricks_won[declarer_team]

    if declarer_tricks >= needed_tricks:
        overtricks = declarer_tricks - needed_tricks
        game_reward = contract_level * 2.0 + overtricks * 0.5
    else:
        undertricks = needed_tricks - declarer_tricks
        game_reward = -(contract_level * 2.0 + undertricks * 3.0)

    # 按隊伍分配 reward
    for t in transitions:
        player_team = 'NS' if t['acting_player'] in ('N', 'S') else 'EW'
        if player_team == declarer_team:
            t['reward'] = game_reward
        else:
            t['reward'] = -game_reward
    # 最後一步標記 done
    if transitions:
        transitions[-1]['done'] = True

    return transitions


# ==========================================
# PPO 更新
# ==========================================
def ppo_update(actor, critic, optimizer_actor, optimizer_critic,
               all_transitions, device):
    """對收集到的所有 transitions 做 PPO 更新"""

    # --- 按玩家分組計算 GAE ---
    # 收集所有 transitions 的 GAE
    # 先按遊戲分組 (用 done 標記切分)
    game_transitions = []
    current_game = []
    for t in all_transitions:
        current_game.append(t)
        if t['done']:
            game_transitions.append(current_game)
            current_game = []

    # 對每局遊戲的每個玩家分別計算 GAE
    processed = []
    for game in game_transitions:
        # 按玩家分組
        player_data = {}
        for t in game:
            p = t['acting_player']
            if p not in player_data:
                player_data[p] = []
            player_data[p].append(t)

        # 對每個玩家計算 GAE
        for p, steps in player_data.items():
            rewards = [s['reward'] for s in steps]
            values = [s['value'] for s in steps]
            # 只有最後一步是 done
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
        # 隨機打亂
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
def evaluate(actor, play_agent, device, n_games=EVAL_GAMES):
    """用目前的 actor 打 n_games 局，回傳平均 reward"""
    env = BridgeGymEnv()
    total_reward = 0.0
    contracts_made = 0
    passed_out = 0

    for _ in range(n_games):
        obs, info = env.reset()

        # 叫牌 (用 actor greedy)
        while not env.playing_phase and not env.game_over:
            state_vec = obs["bidding_state"]
            legal_actions = env.get_legal_bidding_actions()
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = actor(state_tensor)
                mask = torch.full((38,), float('-inf'), device=device)
                for a in legal_actions:
                    mask[a] = 0.0
                logits = logits.squeeze(0) + mask
                action = logits.argmax().item()
            obs, _, _, _, info = env.step_bidding(action)

        if env.contract == "Passed Out":
            passed_out += 1
            total_reward -= 1.0
            continue

        # 打牌 (凍結)
        while not env.game_over:
            legal_indices = env.get_legal_playing_card_indices()
            card_idx, _, _ = play_agent.select_action(obs["playing_state"], legal_indices)
            obs, _, _, _, info = env.step_playing(card_idx)

        level = int(env.contract[0])
        needed = level + 6
        d_team = 'NS' if env.declarer in ('N', 'S') else 'EW'
        d_tricks = env.tricks_won[d_team]
        if d_tricks >= needed:
            contracts_made += 1
            total_reward += level * 2.0
        else:
            total_reward -= (needed - d_tricks) * 3.0

    avg_reward = total_reward / n_games
    made_rate = contracts_made / max(n_games - passed_out, 1)
    return avg_reward, made_rate, passed_out


# ==========================================
# 主訓練迴圈
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 叫牌 RL 訓練 (PPO) | 設備: {device} ===\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 初始化 Actor (從預訓練權重開始) ---
    actor = BiddingModel().to(device)
    bid_model_path = os.path.join(SAVE_DIR, "bidding_model2.pth")
    if os.path.exists(bid_model_path):
        actor.load_state_dict(torch.load(bid_model_path, map_location=device))
        print(f"已載入預訓練叫牌模型: {bid_model_path}")
    else:
        print("警告: 找不到預訓練叫牌模型，從隨機權重開始訓練")
    actor.eval()

    # --- 初始化 Critic ---
    critic = BiddingValueNet().to(device)
    critic.eval()

    # --- 凍結的 PlayAgent ---
    play_agent = PlayAgent(
        model_path=os.path.join(SAVE_DIR, "policy_165dim_latest.pth"),
        device=device
    )

    # --- 優化器 ---
    optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # --- 訓練迴圈 ---
    env = BridgeGymEnv()
    best_reward = float('-inf')
    episode = 0

    pbar = tqdm(total=TOTAL_EPISODES, desc="Bidding RL")
    while episode < TOTAL_EPISODES:
        # 收集 BATCH_GAMES 局的資料
        all_transitions = []
        batch_rewards = []

        for _ in range(BATCH_GAMES):
            transitions = collect_one_game(env, actor, critic, play_agent, device)
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
            avg_reward, made_rate, po = evaluate(actor, play_agent, device)
            print(f"\n[Eval @ {episode}] Avg Reward: {avg_reward:.2f} | "
                  f"合約完成率: {made_rate:.1%} | 流局: {po}/{EVAL_GAMES}")

            # 儲存最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(actor.state_dict(),
                           os.path.join(SAVE_DIR, "bidding_rl_best.pth"))
                print(f"  → 已儲存最佳模型 (reward: {best_reward:.2f})")

            # 總是儲存最新
            torch.save(actor.state_dict(),
                       os.path.join(SAVE_DIR, "bidding_rl_latest.pth"))
            torch.save(critic.state_dict(),
                       os.path.join(SAVE_DIR, "bidding_rl_critic.pth"))

    pbar.close()
    print(f"\n=== 訓練完成! 最佳 reward: {best_reward:.2f} ===")


if __name__ == "__main__":
    main()
