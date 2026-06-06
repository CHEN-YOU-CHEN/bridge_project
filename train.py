import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from Src.model import BridgePolicyNet
import os
from tqdm import tqdm 

def train_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    dataset_path = "Data/processed/bridge_dataset.pt"
    if not os.path.exists(dataset_path):
        print("錯誤：找不到預處理檔案！")
        return

    print("正在載入大型資料集...")
    checkpoint = torch.load(dataset_path)
    

    states = checkpoint['states']   
    actions = checkpoint['actions'] 
    
    print(f"測試資料維度: {states.shape}")


    # 3. 封裝與切分 (90% 訓練, 10% 驗證)
    full_dataset = TensorDataset(states, actions)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 這裡將 Batch Size 提高到 2048，並開啟 num_workers 加速 (Windows 若報錯請設為 0)
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, pin_memory=True)
    print(f"訓練樣本: {train_size}, 驗證樣本: {val_size}")

 
    model = BridgePolicyNet(input_dim=165, hidden_dim=1024, output_dim=52).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 5. 訓練迴圈
    epochs = 15
    best_val_acc = 0.0  # 記錄歷史最高準確率

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_states, batch_actions in pbar:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device).long()

            optimizer.zero_grad()
            outputs = model(batch_states)
            loss = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 每輪結束進行驗證
        model.eval()
        correct = 0
        print("正在執行驗證...")
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device).long()
                
                outputs = model(batch_states)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_actions).sum().item()
        
        accuracy = correct / val_size
        print(f"--- Epoch {epoch+1} 結果: Avg Loss: {total_loss/len(train_loader):.4f}, Val Acc: {accuracy:.4%} ---")

        # 6. 儲存機制優化
        os.makedirs("Data/models", exist_ok=True)
        
        # A. 儲存目前為止表現最好的一次
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            torch.save(model.state_dict(), "Data/models/policy_165dim_best.pth")
            print(f"已儲存最佳模型 (目前最高準確率: {best_val_acc:.4%})")
            
        # B. 每回合都備份一次最新進度，防止當機
        torch.save(model.state_dict(), "Data/models/policy_165dim_latest.pth")

    print(f"\n 訓練全數完成！歷史最佳驗證準確率: {best_val_acc:.4%}")

if __name__ == "__main__":
    train_model()