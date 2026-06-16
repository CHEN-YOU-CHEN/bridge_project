# BridgeAI — 橋牌人工智慧專題

> **中山大學 專題製作**  
> 以深度學習（監督式學習 + PPO 強化學習）訓練能夠完整遊玩橋牌的 AI 代理人。

---

## 目錄

1. [專題簡介](#1-專題簡介)
2. [技術架構概覽](#2-技術架構概覽)
3. [資料流程](#3-資料流程)
4. [訓練流程](#4-訓練流程)
5. [目錄與檔案說明](#5-目錄與檔案說明)
   - [根目錄 — 入口腳本](#根目錄--入口腳本)
   - [Src/ — 核心模組](#src--核心模組)
   - [Data/ — 資料與模型權重](#data--資料與模型權重)
6. [模型架構細節](#6-模型架構細節)
7. [狀態向量設計](#7-狀態向量設計)
8. [環境類別對照](#8-環境類別對照)
9. [檔案相依關係圖](#9-檔案相依關係圖)
10. [快速開始](#10-快速開始)

---

## 1. 專題簡介

橋牌（Contract Bridge）是世界上最複雜的紙牌遊戲之一，分為兩個主要階段：

- **叫牌階段（Bidding）**：四位玩家輪流競標合約，決定王牌花色與目標磴數。
- **打牌階段（Playing / Card Play）**：依合約由莊家（Declarer）與夢家（Dummy）搭檔對抗防守方，競取磴數。

本專題採用雙模型架構：
1. **叫牌模型** — `BiddingModel`：以監督式學習從真實牌譜中學習叫牌策略。
2. **打牌模型** — `BridgePolicyNet`：先以監督式學習預訓練，再以 **PPO（Proximal Policy Optimization）** 強化學習微調，達成最大化合約完成率的目標。

---

## 2. 技術架構概覽

```
BBO 牌譜 (.lin)
    │
    ▼
Src/dataset.py         ← 解析 LIN 格式，生成 165 維狀態向量
    │
    ▼
Data/processed/bridge_dataset.pt   ← 預處理後的訓練集（PyTorch 張量）
    │
    ├──► train.py                  ← 監督式學習，訓練打牌模型
    │        └── BridgePolicyNet   ← 存入 policy_165dim_best.pth
    │
    ├──► Src/bidding_model.py      ← 監督式學習，訓練叫牌模型
    │        └── BiddingModel      ← 存入 bidding_model.pth
    │
    └──► train_playing_rl.py       ← PPO 強化學習，微調打牌模型
             ├── BridgePolicyNet (Actor)   ← playing_rl_best.pth
             └── PlayingValueNet (Critic)  ← playing_rl_critic.pth

最終推論
    ├── bridgeAI.py                ← 完整對局（叫牌 + 打牌雙模型）
    └── predict_action.py          ← 互動式單步推薦工具
```

---

## 3. 資料流程

### 原始資料：BBO LIN 格式
`Data/bbo_data/` 資料夾存放從 **BBO（Bridge Base Online）** 下載的牌譜檔案（`.lin` 格式），每份檔案包含數場完整的橋牌對局記錄，包含發牌資訊（`md|`）、叫牌序列（`mb|`）、每張出牌記錄（`pc|`）、以及王牌資訊（`tr|`）。

### 資料處理：`Src/dataset.py`
`BridgeDataset` 類別負責讀取所有 `.lin` 檔案，並為每一步「出牌動作」生成一個訓練樣本：
- **輸入（State）**：165 維浮點數向量（詳見第 7 節）
- **標籤（Action）**：對應的出牌索引（0~51）

處理完成後，資料以 PyTorch 張量格式儲存至 `Data/processed/bridge_dataset.pt`（約 750MB）。

---

## 4. 訓練流程

### 階段一：監督式預訓練（Imitation Learning）

**打牌模型**（`train.py`）
1. 載入 `bridge_dataset.pt`
2. 90/10 分割為訓練集與驗證集
3. 使用 **CrossEntropyLoss** 訓練 `BridgePolicyNet`（165→1024→1024→512→52）
4. 按驗證準確率儲存最佳模型至 `Data/models/policy_165dim_best.pth`

**叫牌模型**（`Src/bidding_model.py` 中的 `main()`）
- 使用雙分支網路：
  - 手牌分支（52維 → Linear → 128維）
  - 叫牌歷史分支（Embedding + Linear → 128維）
- 合併後輸出 38 種叫牌動作的機率
- 儲存至 `Data/models/bidding_model.pth`

### 階段二：PPO 強化學習微調（`train_playing_rl.py`）

以監督預訓練的 `policy_165dim_latest.pth` 初始化 Actor，加上從頭訓練的 Critic（`PlayingValueNet`），在完整的兩階段橋牌環境（`BridgeGymEnv`）中進行自我對弈：

| 超參數 | 值 |
|---|---|
| 總對局數 | 50,000 |
| 每批局數（Batch） | 64 |
| PPO Epochs | 4 |
| Mini-batch 大小 | 512 |
| 折扣因子 γ | 0.99 |
| GAE λ | 0.95 |
| PPO Clip ε | 0.2 |
| Actor 學習率 | 1e-5 |
| Critic 學習率 | 5e-5 |
| 熵正則化係數 | 0.01 |

**Reward 設計**：
- 每磴結束：`±1.0`（莊家方視角）
- 遊戲結束：完成合約 `+10 + 超磴 × 2`；未完成 `-(10 + 不足磴 × 3)`
- 防守方自動翻轉 reward 方向

---

## 5. 目錄與檔案說明

### 根目錄 — 入口腳本

---

#### `bridgeAI.py` — **完整 AI 對局入口**
最終對外展示用的主程式。整合叫牌模型與打牌模型，完整模擬一局橋牌（叫牌 + 打牌）：
1. 載入 `BridgePolicyNet`（打牌）與 `BiddingModel`（叫牌）
2. 初始化 `BridgeEnv`，發牌
3. 叫牌階段：以 BiddingModel 推理，加上合法動作遮罩選擇最高機率叫牌
4. 打牌階段：以 BridgePolicyNet 推理，加上合法動作遮罩選擇最佳出牌
5. 打印每步決策與信心值，最終輸出得分

**關鍵函式**：
- `env_state_to_bidding_dim(env)` — 將環境狀態轉為 72 維叫牌向量
- `env_state_to_165dim(env, full_history)` — 將環境狀態轉為 165 維打牌向量

---

#### `train.py` — **打牌監督式學習訓練腳本**
訓練 `BridgePolicyNet` 打牌模型：
- 載入 `Data/processed/bridge_dataset.pt`
- 以 CrossEntropyLoss + Adam 訓練 15 個 epoch
- Batch Size 2048，加速訓練
- 同時儲存最佳模型（`policy_165dim_best.pth`）與每輪備份（`policy_165dim_latest.pth`）

**執行方式**：
```bash
python train.py
```

---

#### `train_playing_rl.py` — **打牌 PPO 強化學習訓練腳本**
橋牌 AI 的核心強化學習訓練程式，包含：
- `PlayingValueNet`（Critic 網路，165→512→256→128→1）
- `collect_one_game()` — 執行一局完整遊戲，叫牌用隨機合法策略，打牌用 Actor 收集 trajectory
- `compute_gae()` — 計算廣義優勢估計（GAE）
- `compute_player_reward()` — 依玩家所在隊伍（NS/EW）轉換 reward 方向
- `ppo_update()` — 按 mini-batch 執行 PPO 梯度更新
- `evaluate()` — 每 500 局評估一次，莊家方使用 Actor greedy，防守方使用隨機策略

**執行方式**：
```bash
python train_playing_rl.py
```

---

#### `predict_action.py` — **互動式打牌建議工具**
供真實對局中使用的命令列輔助工具，使用者手動輸入：
- 自己的 13 張手牌
- 夢家手牌（可選）
- 王牌花色
- 每輪新增的歷史出牌與出牌順位

程式組合 165 維特徵向量，以 `BridgePolicyNet` 推理後顯示前 5 名建議出牌及其機率（含合法動作過濾）。

**執行方式**：
```bash
python predict_action.py
```

---

#### `main_random_agent.py` — **隨機代理人測試腳本**
使用純隨機策略完整模擬一局橋牌（叫牌 + 打牌），用於：
- 驗證 `BridgeEnv` 環境的基本功能
- 快速觀察遊戲流程格式

**執行方式**：
```bash
python main_random_agent.py
```

---

#### `run_test.py` — **Gymnasium 環境規範驗證腳本**
使用 `gymnasium.utils.env_checker.check_env()` 驗證 `BridgeEnv` 是否符合 Gymnasium 標準介面，同時進行完整的合法動作隨機對局測試。此腳本在環境大改動後用來確保介面規範未被破壞。

**注意**：腳本中引用了 `env.decode_action()` 與 `DECK` 等可能已過時的介面，主要作為開發期的回歸測試工具。

---

### Src/ — 核心模組

---

#### `Src/bridge_env.py` — **橋牌遊戲環境（最核心模組）**
整個專題最核心的檔案，定義了兩個環境類別：

**`BridgeEnv`（Gymnasium 相容環境）**
- 用於早期開發與簡化測試，**跳過完整叫牌流程**，直接隨機決定莊家與王牌
- 狀態空間：165 維 Dict 觀察（observation + action_mask）
- 動作空間：Discrete(52)，代表出牌
- 實作標準 Gymnasium API：`reset()`、`step()`、`render()`

**`BridgeGymEnv`（完整兩階段環境）**
- 供 `train_playing_rl.py` 使用的完整橋牌環境
- **叫牌觀察**（72 維）：`[手牌 52] + [最近 20 步叫牌歷史]`
- **打牌觀察**（165 維）：`[歷史出牌 52] + [自己手牌 52] + [夢家手牌 52] + [王牌 5] + [順位 4]`
- 完整叫牌規則：Pass、有效叫牌大小限制、Double、Redouble
- 正確莊家判定（`_find_declarer()`）：找出隊伍中最早叫出該花色的玩家
- 打牌 Reward：每磴 ±1.0 + 遊戲結束獎懲
- **資訊保護**：全域首攻時夢家手牌隱藏（防資訊洩漏）

**全局常數**：
- `BIDDING_ACTIONS` — 38 種叫牌動作的名稱列表（Pass, 1C~7N, Double, Redouble）
- `PLAYERS` — `['N', 'E', 'S', 'W']`

---

#### `Src/model.py` — **打牌策略網路**

**`BridgePolicyNet`**（Actor 網路）
```
輸入 (165維)
    → Linear(165, 1024) + BatchNorm + ReLU + Dropout(0.3)
    → Linear(1024, 1024) + ReLU + Dropout(0.2)
    → Linear(1024, 512) + ReLU
    → Linear(512, 52)   ← 52 種出牌動作的 logits
```
- 內建**合法動作遮罩**：從輸入向量 `[52:104]`（自己手牌）取得遮罩，對手中沒有的牌設定 logit 為 `-1e4`
- 同時作為監督式學習的模型（`train.py`）與 RL 的 Actor（`train_playing_rl.py`）

---

#### `Src/bidding_model.py` — **叫牌模型**

**`BiddingModel`**（雙分支神經網路）
```
手牌分支:   Linear(52, 128) + ReLU → 128維
歷史分支:   Embedding(39, 16) → Linear(320, 128) + ReLU → 128維
合併:       Concat → Linear(256, 256) + ReLU + Dropout → Linear(256, 38)
```
- 叫牌歷史以 Embedding 處理，Padding index 為 0（對應原始 -1）
- 輸出 38 種動作的 logits（0=Pass, 1~35=有效叫牌, 36=Double, 37=Redouble）

**`BridgeBiddingDataset`** — 讀取 `bridge_dataset.pt` 中的 72 維叫牌狀態資料。

---

#### `Src/dataset.py` — **資料集處理**

**`BridgeDataset`** 負責解析 BBO LIN 格式牌譜並生成訓練資料：

1. 讀取 `Data/bbo_data/*.lin` 中所有牌譜
2. 解析每局發牌（`md|`），重建四家初始手牌
3. 從 `tr|` 或 `mb|` 取得王牌資訊，建立 5 維 One-hot 向量
4. 解析出牌序列（`pc|`），利用 `determine_trick_winner()` 模擬完整對局
5. 為每步出牌生成 165 維狀態向量（見第 7 節），配對對應動作
6. 所有樣本轉為 PyTorch 張量，儲存至 `Data/processed/bridge_dataset.pt`

**特別處理**：
- 全域首攻（第 1 張牌）時夢家手牌設為全 0（防資訊洩漏）
- 夢家出牌後同步從夢家手牌向量中扣除

---

#### `Src/utils.py` — **通用工具函式**

整個專題的基礎工具層，被多個模組引用：

| 常數/函式 | 說明 |
|---|---|
| `SUITS`, `RANKS` | 花色與點數定義 |
| `TRUMP_MAP` | 花色字元 → 索引的映射（S=0, H=1, D=2, C=3, N=4） |
| `CARD_TO_IDX` | 牌名字串 → 0~51 索引的對照表（如 `'SA' → 12`） |
| `IDX_TO_CARD` | 0~51 索引 → 牌名字串的對照表（反查用） |
| `card_str_to_idx()` | 將 BBO 格式牌名（如 `'HA'`）轉為索引，支援花色在前或在後 |
| `get_suit_from_idx()` | 根據索引回傳花色編號 |
| `get_trump_vec()` | 從 LIN 遊戲內容解析王牌，回傳 5 維 One-hot 向量 |
| `get_legal_mask()` | 根據手牌與引牌花色，計算合法出牌遮罩（52 維布林陣列） |
| `determine_trick_winner()` | 給定 4 張牌與出牌者，依橋牌規則判定磴的贏家（支援王牌） |
| `encode_contract()` | 將合約字串轉為王牌花色索引（輔助功能） |

---

#### `Src/bid_agent.py` — **叫牌代理人封裝**

**`BidAgent`** 類別：封裝 `BiddingModel` 的載入與推理，提供簡潔的 Agent 介面：
- `__init__(model_path, device)` — 載入模型
- `select_action(bidding_state, legal_actions)` — 輸入 72 維叫牌狀態與合法動作列表，回傳 `(action, confidence)`

此類別供外部程式整合使用（如未來的對局伺服器或 GUI）。

---

#### `Src/play_agent.py` — **打牌代理人封裝**

**`PlayAgent`** 類別：封裝 `BridgePolicyNet` 的載入與推理：
- `__init__(model_path, device)` — 載入模型
- `select_action(playing_state, legal_card_indices)` — 輸入 165 維打牌狀態與合法牌索引列表，回傳 `(card_idx, card_name, confidence)`

**注意**：此類別目前引用了 `bridge_gym_env.IDX_TO_CARD`（已整合至 `Src/utils.py`），若單獨使用需確認 import 路徑。

---

#### `Src/checkdata.py` — **資料集驗證工具**

用於在訓練前抽查 `bridge_dataset.pt` 資料品質的除錯工具：
- 隨機抽取 N 筆樣本，還原並顯示手牌、夢家手牌、歷史、王牌、順位
- 驗證「出牌是否在手牌中」與「歷史是否與手牌重疊」等關鍵合法性
- 支援 165 維與 113 維兩種特徵格式（相容舊版資料）

**執行方式**：
```bash
python Src/checkdata.py
```

---

#### `Src/__init__.py` — **套件初始化檔**

空白的 `__init__.py`，使 `Src/` 資料夾成為 Python package，允許其他腳本以 `from Src.xxx import yyy` 的方式引用模組。

---

### Data/ — 資料與模型權重

```
Data/
├── bbo_data/          ← BBO 原始牌譜（LIN 格式，70000~71xxx.lin）
│   └── *.lin
├── processed/         ← 預處理後的訓練集
│   └── bridge_dataset.pt    （~750MB，PyTorch 張量格式）
└── models/            ← 訓練好的模型權重
    ├── bidding_model.pth         （叫牌模型，~500KB）
    ├── policy_165dim_best.pth    （打牌監督預訓練最佳版，~6.8MB）
    ├── policy_165dim_latest.pth  （打牌監督預訓練最新版，~6.8MB）
    ├── playing_rl_best.pth       （PPO 強化學習打牌最佳版，~6.8MB）
    ├── playing_rl_latest.pth     （PPO 強化學習打牌最新版，~6.8MB）
    ├── playing_rl_critic.pth     （PPO Critic 網路，~1MB）
    └── policy_113dim.pth         （舊版 113 維輸入的打牌模型，備份用）
```

> **注意**：`Data/` 資料夾已加入 `.gitignore`，不會上傳至 Git 儲存庫。

---

## 6. 模型架構細節

### BridgePolicyNet（打牌 Actor）

| 層 | 輸入維度 | 輸出維度 | 備註 |
|---|---|---|---|
| Linear | 165 | 1024 | |
| BatchNorm1d | 1024 | 1024 | |
| ReLU + Dropout(0.3) | — | — | |
| Linear | 1024 | 1024 | |
| ReLU + Dropout(0.2) | — | — | |
| Linear | 1024 | 512 | |
| ReLU | — | — | |
| Linear | 512 | 52 | 52 張牌的 logits |
| Legal Mask | — | 52 | 從 x[52:104] 自動遮蔽非手牌 |

### BiddingModel（叫牌）

| 分支 | 層 | 輸入 → 輸出 |
|---|---|---|
| 手牌 | Linear + ReLU | 52 → 128 |
| 歷史 | Embedding(39, 16) | 20 步 → 20×16=320 |
| 歷史 | Linear + ReLU | 320 → 128 |
| 合併 | Concat | 128+128=256 |
| 輸出 | Linear + ReLU + Dropout + Linear | 256 → 256 → 38 |

### PlayingValueNet（PPO Critic）

| 層 | 維度 |
|---|---|
| Linear + ReLU | 165 → 512 |
| Linear + ReLU | 512 → 256 |
| Linear + ReLU | 256 → 128 |
| Linear | 128 → 1 |

---

## 7. 狀態向量設計

### 打牌狀態（165 維）

| 維度範圍 | 名稱 | 說明 |
|---|---|---|
| `[0:52]` | 歷史出牌 | 所有已出牌的 0/1 遮罩 |
| `[52:104]` | 自己手牌 | 目前玩家持有的牌的 0/1 遮罩 |
| `[104:156]` | 夢家手牌 | 夢家目前持有的牌（首磴首攻前為全 0） |
| `[156:161]` | 王牌 | 5 維 One-hot：S/H/D/C/NT |
| `[161:165]` | 順位 | 4 維 One-hot：本磴第幾位出牌（1~4） |

### 叫牌狀態（72 維）

| 維度範圍 | 名稱 | 說明 |
|---|---|---|
| `[0:52]` | 自己手牌 | 目前叫牌者的手牌 0/1 遮罩 |
| `[52:72]` | 叫牌歷史 | 最近 20 步叫牌動作索引，未滿則以 -1 填補 |

---

## 8. 環境類別對照

| 特性 | `BridgeEnv` | `BridgeGymEnv` |
|---|---|---|
| **用途** | 早期測試 / Gymnasium 驗證 | PPO 強化學習訓練 |
| **叫牌** | ❌ 跳過（隨機莊家） | ✅ 完整叫牌規則 |
| **玩家表示** | 整數 0~3 | 字串 'N'/'E'/'S'/'W' |
| **API** | Gymnasium 標準 `step()` | 分離 `step_bidding()` / `step_playing()` |
| **觀察** | `{'observation': 165維, 'action_mask': 52維}` | `{'bidding_state': 72維, 'playing_state': 165維}` |
| **Reward** | 相對磴數（歸一化至 -1~1） | 按磴 ±1.0 + 最終獎懲 |
| **引用自** | `main_random_agent.py`, `run_test.py`, `bridgeAI.py` | `train_playing_rl.py` |

---

## 9. 檔案相依關係圖

```
Src/utils.py
    ↑ 被引用
    ├── Src/bridge_env.py    (determine_trick_winner, CARD_TO_IDX)
    ├── Src/dataset.py       (card_str_to_idx, get_trump_vec, determine_trick_winner)
    ├── Src/checkdata.py     (工具函式)
    └── predict_action.py    (CARD_TO_IDX, IDX_TO_CARD, get_legal_mask)

Src/model.py  (BridgePolicyNet)
    ↑ 被引用
    ├── train.py
    ├── train_playing_rl.py
    ├── bridgeAI.py
    ├── predict_action.py
    └── Src/play_agent.py

Src/bidding_model.py  (BiddingModel)
    ↑ 被引用
    ├── bridgeAI.py
    └── Src/bid_agent.py

Src/bridge_env.py  (BridgeEnv, BridgeGymEnv, BIDDING_ACTIONS)
    ↑ 被引用
    ├── bridgeAI.py              (BridgeEnv, BIDDING_ACTIONS)
    ├── main_random_agent.py     (BridgeEnv, BIDDING_ACTIONS)
    ├── run_test.py              (BridgeEnv)
    └── train_playing_rl.py      (BridgeGymEnv, PLAYERS)

Src/dataset.py  (BridgeDataset)
    ↑ 被引用
    └── (直接執行 python Src/dataset.py 生成 bridge_dataset.pt)

Data/processed/bridge_dataset.pt
    ↑ 被讀取
    ├── train.py
    └── Src/bidding_model.py (BridgeBiddingDataset)

Data/models/*.pth
    ↑ 被讀取
    ├── bridgeAI.py
    ├── predict_action.py
    └── train_playing_rl.py  (載入預訓練後繼續微調)
```

---

## 10. 快速開始

### 環境需求

```bash
Python 3.9+
torch >= 2.0
numpy
gymnasium
tqdm
```

建議使用虛擬環境：
```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install torch numpy gymnasium tqdm
```

### 步驟一：準備資料（BBO 牌譜已存在時跳過）

將 `.lin` 格式的牌譜檔案放入 `Data/bbo_data/`，然後執行：

```bash
python Src/dataset.py
```

這將生成 `Data/processed/bridge_dataset.pt`（約需數分鐘）。

### 步驟二：預訓練打牌模型（監督式學習）

```bash
python train.py
```

完成後生成：
- `Data/models/policy_165dim_best.pth`
- `Data/models/policy_165dim_latest.pth`

### 步驟三：PPO 強化學習微調（可選）

```bash
python train_playing_rl.py
```

完成後生成：
- `Data/models/playing_rl_best.pth`
- `Data/models/playing_rl_critic.pth`

### 步驟四：執行完整 AI 對局

```bash
python bridgeAI.py
```

### 互動式打牌建議

```bash
python predict_action.py
```

### 驗證資料集品質

```bash
python Src/checkdata.py
```

---

## 關於 LIN 格式

BBO 的 `.lin` 格式是橋牌牌譜的標準記錄格式。關鍵欄位：

| 欄位 | 說明 | 範例 |
|---|---|---|
| `qx|` | 局次分隔符 | `qx|o1,` |
| `md|` | 發牌記錄 | `md|3SAJ974H65...` |
| `mb|` | 叫牌序列 | `mb|1S\|mb|P\|` |
| `pc|` | 出牌記錄 | `pc|H5\|pc|H8\|` |
| `tr|` | 王牌花色 | `tr|S` |

---

*本專題為中山大學資訊工程學系專題製作成果。*
