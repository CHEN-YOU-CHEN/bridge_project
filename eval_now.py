import torch
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))
from Src.model import LegacyPolicyNet, BridgePolicyNetResNet
from Src.bridge_env import BridgeGymEnv, BIDDING_ACTIONS
import numpy as np


# ============================================================
# 模型設定對照表（路徑 → 架構類別 + 建構參數）
# ============================================================
MODEL_REGISTRY = {
    "Data/models/policy_165dim_best.pth":          (LegacyPolicyNet,       {"input_dim": 165}),
    "Data/models/policy_165dim_latest.pth":         (LegacyPolicyNet,       {"input_dim": 165}),
    "Data/models/policy_resnet_best.pth":           (BridgePolicyNetResNet, {"input_dim": 165, "hidden_dim": 512, "output_dim": 52, "num_blocks": 4}),
    "Data/models/policy_resnet_latest.pth":         (BridgePolicyNetResNet, {"input_dim": 165, "hidden_dim": 512, "output_dim": 52, "num_blocks": 4}),
    "Data/models/playing_rl_mix_best.pth":          (LegacyPolicyNet,       {"input_dim": 165}),
    "Data/models/playing_rl_mix_latest.pth":        (LegacyPolicyNet,       {"input_dim": 165}),
    "Data/models/playing_rl_pure_best.pth":         (LegacyPolicyNet,       {"input_dim": 165}),
    "Data/models/playing_rl_pure_latest.pth":       (LegacyPolicyNet,       {"input_dim": 165}),
    "Data/models/playing_rl_resnet_mix_best.pth":   (BridgePolicyNetResNet, {"input_dim": 165, "hidden_dim": 512, "output_dim": 52, "num_blocks": 4}),
    "Data/models/playing_rl_resnet_mix_latest.pth": (BridgePolicyNetResNet, {"input_dim": 165, "hidden_dim": 512, "output_dim": 52, "num_blocks": 4}),
    "Data/models/playing_rl_resnet_pure_best.pth":  (BridgePolicyNetResNet, {"input_dim": 165, "hidden_dim": 512, "output_dim": 52, "num_blocks": 4}),
    "Data/models/playing_rl_resnet_pure_latest.pth":(BridgePolicyNetResNet, {"input_dim": 165, "hidden_dim": 512, "output_dim": 52, "num_blocks": 4}),
}


def load_model(model_path: str, device):
    """根據 MODEL_REGISTRY 自動選擇架構並載入模型。"""
    if model_path not in MODEL_REGISTRY:
        print(f"[警告] '{model_path}' 不在 MODEL_REGISTRY 中，嘗試以 LegacyPolicyNet 載入。")
        ModelClass, kwargs = LegacyPolicyNet, {"input_dim": 165}
    else:
        ModelClass, kwargs = MODEL_REGISTRY[model_path]

    model = ModelClass(**kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def heuristic_bid(env):
    n_hand = env.hands['N']
    s_hand = env.hands['S']
    ns_cards = list(n_hand) + list(s_hand)
    suit_counts = {'S': 0, 'H': 0, 'D': 0, 'C': 0}
    suits_order = ['S', 'H', 'D', 'C']
    for card in ns_cards:
        if isinstance(card, int) or isinstance(card, np.integer):
            suit = suits_order[card // 13]
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
    target_bid_str = f"3{best_suit}"
    try:
        return BIDDING_ACTIONS.index(target_bid_str)
    except ValueError:
        return 0


def evaluate(model_path, device, baseline_path=None, n_games=200):
    """
    評估指定模型的出牌表現。
    自動從 MODEL_REGISTRY 識別架構，無需手動指定。
    """
    actor = load_model(model_path, device)
    print(f"[NS Actor]  {model_path}  ({actor.__class__.__name__})")

    baseline = None
    if baseline_path and os.path.exists(baseline_path):
        baseline = load_model(baseline_path, device)
        print(f"[EW Baseline] {baseline_path}  ({baseline.__class__.__name__})")

    env = BridgeGymEnv()
    total_ns_tricks = 0
    valid_games     = 0

    for _ in range(n_games):
        obs, info = env.reset()
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

            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            if curr_team == 'NS':
                with torch.no_grad():
                    logits = actor(state_tensor)
                    mask   = torch.full((52,), -1e9, device=device)
                    for idx in legal_indices:
                        mask[idx] = 0.0
                    logits = logits.squeeze(0) + mask
                    action = logits.argmax().item()
            else:
                if baseline is not None:
                    with torch.no_grad():
                        logits = baseline(state_tensor)
                        mask   = torch.full((52,), -1e9, device=device)
                        for idx in legal_indices:
                            mask[idx] = 0.0
                        logits = logits.squeeze(0) + mask
                        action = logits.argmax().item()
                else:
                    import random
                    action = random.choice(legal_indices)

            obs, _, _, _, info = env.step_playing(action)

        total_ns_tricks += env.tricks_won['NS']

    avg_ns_tricks = total_ns_tricks / max(valid_games, 1)
    return avg_ns_tricks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="評估橋牌出牌策略模型",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="Data/models/playing_rl_mix_best.pth",
        help="NS 陣營模型路徑（自動識別架構，預設: playing_rl_mix_best.pth）"
    )
    parser.add_argument(
        "--baseline",
        default="Data/models/policy_165dim_best.pth",
        help="EW 陣營對手模型路徑（預設: policy_165dim_best.pth）"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="評估局數（預設: 200）"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"評估中... (使用設備: {device}, 局數: {args.games})")
    avg = evaluate(args.model, device, baseline_path=args.baseline, n_games=args.games)
    print(f"\nNS 平均取得磴數: {avg:.2f} / 13.00")
