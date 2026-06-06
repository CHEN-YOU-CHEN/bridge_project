"""
train_playing_rl.py — PPO 強化學習訓練出牌模型

流程:
  1. 叫牌階段: 使用隨機合法叫牌策略 (不依賴外部模型)
  2. 打牌階段: 使用訓練中的 BridgePolicyNet (Actor) + PlayingValueNet (Critic)
  3. 每磴結束 reward ±1.0，遊戲結束 reward ±10 (超磴×2 / 不足磴×3)
  4. 每 BATCH_GAMES 局做一次 PPO 更新

起點: 從牌譜監督預訓練的 policy_165dim_latest.pth 載入 Actor 權重
"""

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
from Src.bridge_env import BridgeGymEnv, PLAYERS
from Src.model import BridgePolicyNet

# ==========================================
# 超參數
# ==========================================
TOTAL_EPISODES  = 50000   # 總自我對弈局數
BATCH_GAMES     = 64      # 每次收集幾局後做 PPO 更新
PPO_EPOCHS      = 4       # 每次更新重複幾個 epoch
MINI_BATCH_SIZE = 512     # PPO mini-batch 大小
GAMMA           = 0.99    # 折扣因子
GAE_LAMBDA      = 0.95    # GAE λ
CLIP_EPSILON    = 0.2     # PPO clip 範圍
LR_ACTOR        = 1e-5    # Actor 學習率 (微調，比監督學習更小)
LR_CRITIC       = 5e-5    # Critic 學習率
ENTROPY_COEFF   = 0.01    # 熵正則化係數（鼓勵探索）
EVAL_INTERVAL   = 500     # 每幾局做一次評估
EVAL_GAMES      = 50      # 評估時打幾局
SAVE_DIR        = "Data/models"


# ==========================================
# Critic 網路 (估計打牌狀態價值)
# ==========================================
class PlayingValueNet(nn.Module):
    """Critic: 估計 165 維打牌觀察的狀態價值 V(s)"""
    def __init__(self, input_dim=165):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ==========================================
# 隨機合法叫牌策略 (取代外部 BidAgent)
# ==========================================
def random_bid(env):
    """從環境的合法叫牌動作中隨機選一個"""
    legal = env.get_legal_bidding_actions()
    return random.choice(legal) if legal else 0


# ==========================================
# GAE 計算
# ==========================================
def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    """計算 Generalized Advantage Estimation (GAE)"""
    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if (t == len(rewards) - 1 or dones[t]) else values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * (1 - float(dones[t])) * gae
        advantages.insert(0, gae)
    advantages = np.array(advantages, dtype=np.float32)
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ==========================================
# Reward 轉換 (站在出牌者的隊伍角度)
# ==========================================
def compute_player_reward(env_reward, acting_player, declarer):
    """
    環境 reward 是站在莊家方的角度計算。
    防守方玩家需要反轉 reward，讓每方都有正確的學習方向。
    """
    if declarer is None:
        return 0.0
    declarer_team = 'NS' if declarer in ('N', 'S') else 'EW'
    player_team   = 'NS' if acting_player in ('N', 'S') else 'EW'
    return env_reward if player_team == declarer_team else -env_reward


# ==========================================
# 收集一局出牌資料
# ==========================================
def collect_one_game(env, actor, critic, device):
    """
    執行一局完整遊戲，收集打牌階段的 trajectory。
    叫牌階段使用隨機合法策略。

    回傳:
        transitions: list of dict (打牌 transitions)
        game_info: dict (局結資訊)
    """
    obs, info = env.reset()

    # ----- 叫牌階段：隨機合法策略 -----
    while not env.playing_phase and not env.game_over:
        action = random_bid(env)
        obs, _, _, _, info = env.step_bidding(action)

    if env.contract == "Passed Out":
        return [], {'passed_out': True, 'contract': 'Passed Out'}

    declarer = env.declarer

    # ----- 打牌階段：收集 PPO 訓練資料 -----
    transitions = []
    trick_buffer = []

    while not env.game_over:
        acting_player = env.get_current_player()
        state_vec     = obs["playing_state"]              # (165,)
        legal_indices = env.get_legal_playing_card_indices()  # list[int] 0~51

        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = actor(state_tensor)                  # (1, 52)

            # 合法動作遮罩：非法牌設為極小值
            mask = torch.full((52,), -1e9, device=device)
            for idx in legal_indices:
                mask[idx] = 0.0
            masked_logits = logits.squeeze(0) + mask      # (52,)

            dist    = torch.distributions.Categorical(logits=masked_logits)
            action  = dist.sample()
            log_prob = dist.log_prob(action)

            value = critic(state_tensor)                  # (1,)

        card_idx = action.item()

        trick_buffer.append({
            'state':        state_vec.copy(),
            'action':       card_idx,
            'log_prob':     log_prob.item(),
            'value':        value.item(),
            'acting_player': acting_player,
            'legal_mask':   mask.cpu().numpy().copy(),    # (52,) -1e9 / 0
            'reward':       0.0,                          # 佔位，一磴結束時分配
            'done':         False,
        })

        # 執行出牌
        obs, env_reward, terminated, truncated, info = env.step_playing(card_idx)

        # 檢查一磴是否結束（滿 4 張牌即結束）
        if len(trick_buffer) == 4:
            for t in trick_buffer:
                # env_reward 是整隊的回報，分配給該磴所有玩家
                t['reward'] = compute_player_reward(env_reward, t['acting_player'], declarer)
                t['done']   = env.game_over
                transitions.append(t)
            trick_buffer = []

    game_info = {
        'passed_out':  False,
        'contract':    env.contract,
        'declarer':    declarer,
        'tricks_won':  dict(env.tricks_won),
    }
    return transitions, game_info


# ==========================================
# PPO 更新
# ==========================================
def ppo_update(actor, critic, optimizer_actor, optimizer_critic,
               all_transitions, device):
    """對收集到的所有 transitions 做 PPO 更新"""

    # --- 按局遊戲分組，對每位玩家分別計算 GAE ---
    game_transitions = []
    current_game = []
    for t in all_transitions:
        current_game.append(t)
        if t['done']:
            game_transitions.append(current_game)
            current_game = []

    processed = []
    for game in game_transitions:
        # 按玩家分組
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

    # --- 轉為 tensor ---
    states       = torch.FloatTensor(np.array([t['state']    for t in processed])).to(device)
    actions      = torch.LongTensor([t['action']             for t in processed]).to(device)
    old_log_probs = torch.FloatTensor([t['log_prob']         for t in processed]).to(device)
    advantages_t  = torch.FloatTensor([t['advantage']        for t in processed]).to(device)
    returns_t     = torch.FloatTensor([t['return']           for t in processed]).to(device)
    legal_masks   = torch.FloatTensor(np.array([t['legal_mask'] for t in processed])).to(device)

    # Advantage 正規化
    if len(advantages_t) > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    # --- PPO epochs ---
    total_actor_loss  = 0.0
    total_critic_loss = 0.0
    n_updates = 0

    for _ in range(PPO_EPOCHS):
        indices = torch.randperm(len(processed))
        for start in range(0, len(processed), MINI_BATCH_SIZE):
            end    = min(start + MINI_BATCH_SIZE, len(processed))
            mb_idx = indices[start:end]

            mb_states   = states[mb_idx]
            mb_actions  = actions[mb_idx]
            mb_old_lp   = old_log_probs[mb_idx]
            mb_adv      = advantages_t[mb_idx]
            mb_ret      = returns_t[mb_idx]
            mb_masks    = legal_masks[mb_idx]

            # --- Actor 更新 ---
            actor.train()
            logits        = actor(mb_states)               # (B, 52)
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

            # --- Critic 更新 ---
            critic.train()
            values      = critic(mb_states)                # (B,)
            critic_loss = F.mse_loss(values, mb_ret)

            optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer_critic.step()

            total_actor_loss  += actor_loss.item()
            total_critic_loss += critic_loss.item()
            n_updates += 1

    actor.eval()
    critic.eval()
    return (total_actor_loss  / max(n_updates, 1),
            total_critic_loss / max(n_updates, 1))


# ==========================================
# 評估
# ==========================================
def evaluate(actor, device, n_games=EVAL_GAMES):
    """用目前的 actor greedy 打 n_games 局，回傳平均表現"""
    env = BridgeGymEnv()
    total_reward  = 0.0
    contracts_made = 0
    total_tricks   = 0
    valid_games    = 0

    actor.eval()
    for _ in range(n_games):
        obs, info = env.reset()

        # 叫牌 (隨機)
        while not env.playing_phase and not env.game_over:
            obs, _, _, _, info = env.step_bidding(random_bid(env))

        if env.contract == "Passed Out":
            continue

        declarer    = env.declarer
        valid_games += 1

        # 打牌 (Actor 為莊家方 vs Random 為防守方，藉此測量模型學習的絕對強度)
        declarer_team = 'NS' if declarer in ('N', 'S') else 'EW'

        while not env.game_over:
            state_vec     = obs["playing_state"]
            legal_indices = env.get_legal_playing_card_indices()
            
            curr_player = env.get_current_player()
            curr_team   = 'NS' if curr_player in ('N', 'S') else 'EW'

            if curr_team == declarer_team:
                # 莊家方使用 actor
                state_tensor  = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = actor(state_tensor)
                    mask   = torch.full((52,), -1e9, device=device)
                    for idx in legal_indices:
                        mask[idx] = 0.0
                    logits = logits.squeeze(0) + mask
                    action = logits.argmax().item()
            else:
                # 防守方使用隨機策略 (這能避免自我對弈時永遠拿到 6.5 磴的假象)
                action = random.choice(legal_indices)

            obs, _, _, _, info = env.step_playing(action)

        level    = int(env.contract[0])
        needed   = level + 6
        d_team   = 'NS' if declarer in ('N', 'S') else 'EW'
        d_tricks = env.tricks_won[d_team]
        total_tricks += d_tricks

        if d_tricks >= needed:
            contracts_made += 1
            total_reward   += 10.0 + (d_tricks - needed) * 2.0
        else:
            total_reward   -= 10.0 + (needed - d_tricks) * 3.0

    avg_reward  = total_reward  / max(valid_games, 1)
    made_rate   = contracts_made / max(valid_games, 1)
    avg_tricks  = total_tricks  / max(valid_games, 1)
    return avg_reward, made_rate, avg_tricks, valid_games


# ==========================================
# 主訓練迴圈
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 打牌 RL 訓練 (PPO) | 設備: {device} ===\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Actor：從監督預訓練權重載入 ---
    actor = BridgePolicyNet(input_dim=165, hidden_dim=1024, output_dim=52).to(device)
    play_model_path = os.path.join(SAVE_DIR, "policy_165dim_latest.pth")
    if os.path.exists(play_model_path):
        actor.load_state_dict(torch.load(play_model_path, map_location=device))
        print(f"已載入預訓練打牌模型: {play_model_path}")
    else:
        print("警告: 找不到預訓練模型，從隨機權重開始訓練")
    actor.eval()

    # --- Critic：從頭訓練 ---
    critic = PlayingValueNet(input_dim=165).to(device)
    critic.eval()

    # --- 優化器 ---
    optimizer_actor  = optim.Adam(actor.parameters(),  lr=LR_ACTOR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # --- 訓練迴圈 ---
    env          = BridgeGymEnv()
    best_reward  = float('-inf')
    episode      = 0

    pbar = tqdm(total=TOTAL_EPISODES, desc="Playing RL (PPO)")
    while episode < TOTAL_EPISODES:

        # 收集 BATCH_GAMES 局資料
        all_transitions = []
        batch_rewards   = []

        for _ in range(BATCH_GAMES):
            transitions, game_info = collect_one_game(env, actor, critic, device)
            all_transitions.extend(transitions)
            if transitions:
                # 取得該局莊家隊伍
                declarer = game_info.get('declarer')
                declarer_team = 'NS' if declarer in ('N', 'S') else 'EW'
                # 只加總莊家隊伍視角的 reward，因為兩隊 reward 會互為正負抵消為 0
                dec_rewards = [t['reward'] for t in transitions if ('NS' if t['acting_player'] in ('N', 'S') else 'EW') == declarer_team]
                game_reward = sum(dec_rewards) if dec_rewards else 0.0
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
            reward  = f"{avg_batch_reward:.2f}",
            a_loss  = f"{actor_loss:.4f}",
            c_loss  = f"{critic_loss:.4f}",
        )

        # 定期評估與儲存
        if episode % EVAL_INTERVAL < BATCH_GAMES:
            avg_reward, made_rate, avg_tricks, valid = evaluate(actor, device)
            print(f"\n[Eval @ ep {episode}] "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"合約完成率: {made_rate:.1%} | "
                  f"平均磴數: {avg_tricks:.1f} | "
                  f"有效局數: {valid}/{EVAL_GAMES}")

            # 儲存最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(actor.state_dict(),
                           os.path.join(SAVE_DIR, "playing_rl_best.pth"))
                print(f"  → 已儲存最佳模型 (reward: {best_reward:.2f})")

            # 每次都備份最新
            torch.save(actor.state_dict(),
                       os.path.join(SAVE_DIR, "playing_rl_latest.pth"))
            torch.save(critic.state_dict(),
                       os.path.join(SAVE_DIR, "playing_rl_critic.pth"))

    pbar.close()
    print(f"\n=== 訓練完成! 最佳 reward: {best_reward:.2f} ===")


if __name__ == "__main__":
    main()
