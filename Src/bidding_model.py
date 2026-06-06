import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# ==========================================
# 1. 定義 Dataset (確保與儲存格式一致)
# ==========================================
class BridgeBiddingDataset(Dataset):
    def __init__(self, processed_path):
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"找不到預處理檔案: {processed_path}")
        
        print(f"載入資料中: {processed_path}")
        data = torch.load(processed_path)
        self.states = data['states']   # 形狀: [N, 72]
        self.actions = data['actions'] # 形狀: [N]

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        # 將狀態拆解為 手牌(52) 與 叫牌歷史(20)
        # 由於歷史叫牌中有 -1 (Padding)，模型輸入前需要處理
        return self.states[idx], self.actions[idx]
    

class BiddingModel(nn.Module):
    def __init__(self, history_len=20, num_bid_types=39): 
        # num_bid_types=39 是因為 0-37 種叫牌 + 1個 Padding(用來處理-1)
        super(BiddingModel, self).__init__()
        
        # 手牌分支 (處理前 52 維)
        self.hand_layer = nn.Sequential(
            nn.Linear(52, 128),
            nn.ReLU()
        )
        
        # 叫牌歷史分支 (處理後 20 維)
        # 我們將索引 +1，讓 -1 變成 0 (作為 Padding Index)
        self.history_embedding = nn.Embedding(num_bid_types, 16, padding_idx=0)
        self.history_layer = nn.Linear(history_len * 16, 128)
        
        # 合併層
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 38) # 最終預測 38 種叫牌動作
        )

    def forward(self, x):
        # 分離手牌與歷史
        hand = x[:, :52]
        history = x[:, 52:].long()
        
        # 處理 Padding: 將歷史索引全部 +1 (-1 變成 0, 0 變成 1...)
        history = history + 1
        
        # 特徵提取
        h_feat = self.hand_layer(hand)
        
        e_feat = self.history_embedding(history) # [batch, 20, 16]
        e_feat = e_feat.view(e_feat.size(0), -1) # 展平 [batch, 320]
        e_feat = F.relu(self.history_layer(e_feat))
        
        # 合併並輸出
        combined = torch.cat([h_feat, e_feat], dim=1)
        return self.fc(combined)

# ==========================================
# 3. 主訓練程式
# ==========================================
def main():
    # 設定
    SAVE_PATH = "Data/processed/bridge_dataset.pt"
    BATCH_SIZE = 128
    EPOCHS = 5
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {DEVICE}")

    # A. 載入資料
    try:
        dataset = BridgeBiddingDataset(SAVE_PATH)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        print(f"載入失敗: {e}")
        return

    # B. 初始化模型、損失函數與優化器
    model = BiddingModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # C. 訓練迴圈
    print("開始訓練...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for states, actions in train_loader:
            states, actions = states.to(DEVICE), actions.to(DEVICE)
            
            # 前向傳播
            outputs = model(states)
            loss = criterion(outputs, actions)
            
            # 計算準確率
            _, predicted = torch.max(outputs.data, 1)
            total += actions.size(0)
            correct += (predicted == actions).sum().item()
            
            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Acc: {acc:.2f}%")

    # D. 儲存訓練好的模型
    torch.save(model.state_dict(), "bidding_model.pth")
    print("模型已儲存為 bidding_model.pth")
    
if __name__ == "__main__":
    main()