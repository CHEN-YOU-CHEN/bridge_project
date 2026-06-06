import numpy as np
import gymnasium as gym
from Src.bridge_env import BridgeEnv, BIDDING_ACTIONS, DECK
from gymnasium.utils.env_checker import check_env

def main():
    # 1. 建立環境
    env = BridgeEnv()

    print("--- 執行 Gymnasium 規範檢查 ---")
    try:
        check_env(env)
        print("檢查通過！\n")
    except Exception as e:
        print(f"檢查失敗 (通常是隨機性或回傳格式問題): {e}")
        # 即使檢查失敗，我們仍嘗試跑完流程以利 Debug
    
    print("--- 開始完整遊戲模擬 (Legal Action Sampling) ---")
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    # 為了方便觀察，我們手動追蹤目前是第幾磴
    current_trick_num = 1

    while not done:
        # 2. 獲取合法動作遮罩
        # mask 是一個 90 維的陣列，1 代表合法，0 代表非法
        mask = env.get_legal_actions()
        
        # 3. 從合法索引中隨機挑選一個
        legal_indices = np.where(mask == 1)[0]
        if len(legal_indices) == 0:
            print("錯誤：找不到任何合法動作！")
            break
            
        action = np.random.choice(legal_indices)
        
        # 記錄執行前的狀態
        current_player = env.get_current_player()
        is_bidding = not env.playing_phase
        
        # 4. 執行動作
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1

        # 5. 印出詳細日誌
        action_type, action_id = env.decode_action(action)
        if action_type == "bid":
            print(f"[叫牌] 步數 {step_count:02d} | 玩家 {current_player} | 動作: {BIDDING_ACTIONS[action_id]:<4}")
        else:
            card_name = DECK[action_id]
            print(f"[打牌] 步數 {step_count:02d} | 玩家 {current_player} | 出牌: {card_name}")
            
            # 偵測一磴結束
            if len(env.current_trick) == 0 and not done:
                print(f"--- 第 {current_trick_num} 磴結束，目前比分 NS:{env.tricks_won['NS']} EW:{env.tricks_won['EW']} ---")
                current_trick_num += 1

    # 6. 遊戲結束總結
    print("\n" + "="*40)
    print(" 遊戲結束報告 ")
    print("="*40)
    print(f"最終合約: {env.contract}")
    print(f"莊家: {env.declarer}")
    print(f"身價狀態: {['None', 'NS', 'EW', 'Both'][env.vulnerability]}")
    print(f"最終比分: NS {env.tricks_won['NS']} 磴 | EW {env.tricks_won['EW']} 磴")
    print(f"總步數: {step_count}")
    print(f"最終 Reward (歸一化得分): {reward:.2f}")
    print("="*40)

if __name__ == "__main__":
    main()