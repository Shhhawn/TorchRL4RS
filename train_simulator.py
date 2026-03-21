import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split


from dataset import RL4RSSimulatorDataset
from nets import LSTMSlateSimulator
from config import cfg
from utils import EarlyStopping

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("初始化 DataLoader...")
    dataset = RL4RSSimulatorDataset(cfg.train_path, max_seq_len=cfg.max_seq_len)
    train_size = int(cfg.sim_train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"数据切分完成 -> 训练集: {train_size} 条, 验证集: {val_size} 条")
    
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.sim_batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.sim_batch_size, shuffle=False, drop_last=False)


    print("初始化模型...")
    model = LSTMSlateSimulator(
        item_vocab_size=400000,
        embed_dim=cfg.sim_embed_dim,
        portrait_dim=cfg.portrait_dim,
        item_feat_dim=cfg.item_feat_dim,
        slate_size=cfg.slate_size
    ).to(device)

    print("定义损失函数和优化器")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.sim_learning_rate, weight_decay=1e-4)

    save_path = f"{cfg.checkpoints_path}/simulator_lstm_best.pth"
    early_stopping = EarlyStopping(
        patience=5, 
        mode='min', 
        filepath=save_path, 
        verbose=True
    )

    epochs = cfg.sim_epoch
    print(f"开始训练，总轮数{epochs}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch: {epoch+1}/{epochs}")

        for batch in progress_bar:
            user_seq = batch["user_seq"].to(device)
            user_portrait = batch["user_portrait"].to(device)
            action_ids = batch["action_ids"].to(device)
            item_feats = batch["item_feats"].to(device)
            labels = batch["labels"].to(device)

            _, _, logits = model(user_seq, user_portrait, item_feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():4f}"})

        avg_train_loss = total_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", colour='green')
            for batch in val_bar:
                user_seq = batch["user_seq"].to(device)
                user_portrait = batch["user_portrait"].to(device)
                item_feats = batch["item_feats"].to(device)
                labels = batch["labels"].to(device)

                _, _, logits = model(user_seq, user_portrait, item_feats)
                val_loss = criterion(logits, labels)
                total_val_loss += val_loss.item()
                val_bar.set_postfix({'val_loss': f"{val_loss.item():.4f}"})

        avg_val_loss = total_val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch+1} 总结 -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if early_stopping(avg_val_loss, model):
            print(f"触发早停机制！模型在第 {epoch+1} 个 Epoch 提前结束训练。")
            break
        print("-" * 60)

    # 6. 保存训练好的“模拟器大脑”
    torch.save(model.state_dict(), save_path)
    print(f"训练结束！模型权重已保存至: {save_path}")

if __name__ == "__main__":
    train()