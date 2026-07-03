import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from Src.model import BridgePolicyNet, BridgePolicyNetResNet
import os
import argparse
from tqdm import tqdm


def train_model(model_type: str = "legacy"):
    """
    監督學習訓練出牌策略模型。

    model_type:
        'legacy'  — 使用 BridgePolicyNet (MLP, hidden_dim=1024)，
                    輸出至 policy_165dim_best.pth
        'resnet'  — 使用 BridgePolicyNetResNet (4 個殘差模塊)，
                    輸出至 policy_resnet_best.pth
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    print(f"模型架構: {'BridgePolicyNet (Legacy MLP)' if model_type == 'legacy' else 'BridgePolicyNetResNet (ResNet)'}\n")

    dataset_path = "Data/processed/bridge_dataset.pt"
    if not os.path.exists(dataset_path):
        print("錯誤：找不到預處理檔案！請先執行資料預處理步驟。")
        return

    print("正在載入大型資料集...")
    checkpoint = torch.load(dataset_path)

    states  = checkpoint['states']
    actions = checkpoint['actions']

    print(f"資料集維度: states={states.shape}, actions={actions.shape}")

    # 封裝與切分 (90% 訓練, 10% 驗證)
    full_dataset = TensorDataset(states, actions)
    train_size   = int(0.9 * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=2048, shuffle=False, pin_memory=True)
    print(f"訓練樣本: {train_size}, 驗證樣本: {val_size}\n")

    # 根據 model_type 選擇模型架構與存檔路徑
    os.makedirs("Data/models", exist_ok=True)
    if model_type == "resnet":
        model       = BridgePolicyNetResNet(input_dim=165, hidden_dim=512, output_dim=52, num_blocks=4).to(device)
        path_best   = "Data/models/policy_resnet_best.pth"
        path_latest = "Data/models/policy_resnet_latest.pth"
    else:
        model       = BridgePolicyNet(input_dim=165, hidden_dim=1024, output_dim=52).to(device)
        path_best   = "Data/models/policy_165dim_best.pth"
        path_latest = "Data/models/policy_165dim_latest.pth"

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs       = 15
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_states, batch_actions in pbar:
            batch_states  = batch_states.to(device)
            batch_actions = batch_actions.to(device).long()

            optimizer.zero_grad()
            outputs = model(batch_states)
            loss    = criterion(outputs, batch_actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 驗證
        model.eval()
        correct = 0
        print("正在執行驗證...")
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states  = batch_states.to(device)
                batch_actions = batch_actions.to(device).long()

                outputs = model(batch_states)
                preds   = outputs.argmax(dim=1)
                correct += (preds == batch_actions).sum().item()

        accuracy = correct / val_size
        avg_loss = total_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} 結果: Avg Loss: {avg_loss:.4f}, Val Acc: {accuracy:.4%} ---")

        # 儲存機制
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            torch.save(model.state_dict(), path_best)
            print(f"已儲存最佳模型 → {path_best}  (目前最高準確率: {best_val_acc:.4%})")

        torch.save(model.state_dict(), path_latest)

    print(f"\n訓練全數完成！歷史最佳驗證準確率: {best_val_acc:.4%}")
    print(f"最佳模型已儲存至: {path_best}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="橋牌出牌策略監督學習訓練腳本")
    parser.add_argument(
        "--model",
        choices=["legacy", "resnet"],
        default="legacy",
        help=(
            "legacy: 使用 BridgePolicyNet (MLP, hidden=1024)，存至 policy_165dim_best.pth (預設)\n"
            "resnet: 使用 BridgePolicyNetResNet (4 個殘差模塊)，存至 policy_resnet_best.pth"
        )
    )
    args = parser.parse_args()
    train_model(model_type=args.model)