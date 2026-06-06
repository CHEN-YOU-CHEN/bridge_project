"""
Runner - 橋牌 AI 遊戲主迴圈

負責：
  1. 初始化環境 (BridgeGymEnv)
  2. 初始化代理 (BidAgent, PlayAgent)
  3. 執行完整的一局遊戲 (叫牌 → 打牌 → 結算)
"""

import torch
from Src.bridge_gym_env import BridgeGymEnv, BIDDING_ACTIONS, IDX_TO_CARD, PLAYERS
from Src.bid_agent import BidAgent
from Src.play_agent import PlayAgent


def run_game(bid_agent, play_agent, env=None, verbose=True):
    """
    執行一局完整的橋牌遊戲。

    參數:
        bid_agent: BidAgent 實例
        play_agent: PlayAgent 實例
        env: BridgeGymEnv 實例 (若為 None 則自動建立)
        verbose: 是否印出遊戲過程

    回傳:
        dict: 遊戲結果，包含合約、莊家、磴數等資訊
    """
    if env is None:
        env = BridgeGymEnv()

    obs, info = env.reset()
    initial_hands = {p: env.hands[p].copy() for p in PLAYERS}

    if verbose:
        print("所有人手牌已發好。")
        for p in PLAYERS:
            print(f"  {p} 手牌: {initial_hands[p]}")
        print("-" * 40)

    # =========================
    # 叫牌階段
    # =========================
    if verbose:
        print("\n=== 開始叫牌階段 ===")
        print(f"**本局由發牌者 (Dealer) {env.get_current_player()} 先開始叫牌**")

    while not env.playing_phase and not env.game_over:
        current_player = env.get_current_player()
        legal_actions = env.get_legal_bidding_actions()

        action, confidence = bid_agent.select_action(
            obs["bidding_state"], legal_actions
        )

        if verbose:
            action_name = BIDDING_ACTIONS[action]
            print(f"  [{current_player}] 叫牌: {action_name}")

        obs, reward, terminated, truncated, info = env.step_bidding(action)

    if verbose:
        print("\n=== 叫牌結束 ===")
        print(f"最終合約: {env.contract}")

    if env.contract == "Passed Out":
        if verbose:
            print("四家 Pass，流局！")
        return {
            "contract": "Passed Out",
            "declarer": None,
            "tricks_won": dict(env.tricks_won),
            "passed_out": True,
        }

    if verbose:
        print(f"莊家 (Declarer): {env.declarer}")
        print("-" * 40)

    # =========================
    # 打牌階段
    # =========================
    if verbose:
        print("\n=== 開始打牌階段 ===")

    trick_count = 1

    while not env.game_over:
        if len(env.current_trick) == 0 and verbose:
            print(f"-- 第 {trick_count} 磴開始 -- "
                  f"(目前比分 NS:{env.tricks_won['NS']} EW:{env.tricks_won['EW']})")

        current_player = env.get_current_player()
        legal_card_indices = env.get_legal_playing_card_indices()

        card_idx, card_name, confidence = play_agent.select_action(
            obs["playing_state"], legal_card_indices
        )

        if verbose:
            remaining = len(env.hands[current_player]) - 1
            print(f"  [{current_player}] 出牌: {card_name} "
                  f"(AI 信心: {confidence*100:.1f}%, 尚餘 {remaining} 張)")

        obs, reward, terminated, truncated, info = env.step_playing(card_idx)

        if len(env.current_trick) == 0 and not env.game_over:
            trick_count += 1
            if verbose:
                print()

    # =========================
    # 結算
    # =========================
    result = {
        "contract": env.contract,
        "declarer": env.declarer,
        "tricks_won": dict(env.tricks_won),
        "passed_out": False,
    }

    if verbose:
        print("\n=== 遊戲結束 ===")
        print(f"最終合約: {result['contract']} | 莊家: {result['declarer']}")
        print(f"取得磴數 -> 南北(NS): {result['tricks_won']['NS']}, "
              f"東西(EW): {result['tricks_won']['EW']}")

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 啟動 AI 代理人 (使用設備: {device}) ===\n")

    # 初始化代理
    bid_agent = BidAgent(
        model_path="Data/models/bidding_model2.pth",
        device=device,
    )
    play_agent = PlayAgent(
        model_path="Data/models/policy_165dim_best.pth",
        device=device,
    )

    # 初始化環境
    env = BridgeGymEnv()

    # 執行一局
    print("\n" + "=" * 50)
    print("=== 初始化橋牌環境 ===")
    result = run_game(bid_agent, play_agent, env, verbose=True)


if __name__ == "__main__":
    main()
