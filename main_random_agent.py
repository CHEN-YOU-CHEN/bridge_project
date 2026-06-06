import random
from Src.bridge_env import BridgeEnv, BIDDING_ACTIONS

def main():
    print("=== 初始化橋牌環境 ===")
    env = BridgeEnv()
    state = env.reset()
    
    # 記錄各家初始手牌用來印出
    initial_hands = {p: env.hands[p].copy() for p in ['N', 'E', 'S', 'W']}
    print("所有人手牌已發好。")
    for p in ['N', 'E', 'S', 'W']:
        print(f"{p} 手牌: {initial_hands[p]}")
    print("-" * 30)
    
    print("\n=== 開始隨機叫牌階段 ===")
    print(f"**本局由發牌者 (Dealer) {state['player']} 先開始叫牌**")
    
    while not state['playing_phase'] and not env.game_over:
        current_player = state['player']
        legal_actions = env.get_legal_bidding_actions()
        
        # 50% 機率直接 Pass, 50% 機率隨機喊牌
        if random.random() < 0.5 or len(legal_actions) == 1:
            action = 0
        else:
            action = random.choice(legal_actions[1:])
            
        action_name = BIDDING_ACTIONS[action]
        print(f"[{current_player}] 叫牌: {action_name}")
        
        state, _, _, _ = env.step(action)

    print("\n=== 叫牌結束 ===")
    print(f"最終合約: {state['contract']}")
    
    if state['contract'] == "Passed Out":
        print("四家 Pass，流局！")
        return

    print(f"莊家 (Declarer): {state['declarer']}")
    print("-" * 30)
    
    print("\n=== 開始隨機打牌階段 ===")
    trick_count = 1
    
    while not env.game_over:
        if len(env.current_trick) == 0:
            print(f"-- 第 {trick_count} 磴開始 -- (目前比分 NS:{state['tricks_won']['NS']} EW:{state['tricks_won']['EW']})")
            
        current_player = state['player']
        legal_cards = env.get_legal_playing_actions()
        
        # 隨機挑一張合法的牌打出
        card_to_play = random.choice(legal_cards)
        print(f"[{current_player}] 出牌: {card_to_play} (尚餘 {len(env.hands[current_player])} 張)")
        
        state, _, done, _ = env.step(card_to_play)
        
        if len(env.current_trick) == 0 and not done:
            # 代表剛結算完一磴
            trick_count += 1
            print()

    print("\n=== 遊戲結束 ===")
    print(f"最終合約: {state['contract']} | 莊家: {state['declarer']}")
    print(f"取得磴數 -> 南北(NS): {state['tricks_won']['NS']}, 東西(EW): {state['tricks_won']['EW']}")

if __name__ == "__main__":
    main()
