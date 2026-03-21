import torch
import torch.nn as nn

class LSTMSlateSimulator(nn.Module):
    def __init__(self, 
                 item_vocab_size, 
                 embed_dim=32, 
                 portrait_dim=42, 
                 item_feat_dim=40, 
                 slate_size=9,
                 dropout_rate=0.3
                 ):
        """
        item_vocab_size: 物品的总数（用于 Embedding 查表）
        embed_dim: 序列中物品 ID 的嵌入维度
        portrait_dim: user_portrait 的特征维度
        item_feat_dim: item_feature 每段的特征维度
        slate_size: 一次推荐的商品数量
        """
        super().__init__()
        
        # ==========================================
        # 1. 动态兴趣提取器 (处理历史序列)
        # ==========================================
        # padding_idx=0 保证填充位不参与梯度更新
        self.item_embedding = nn.Embedding(num_embeddings=item_vocab_size,
                                           embedding_dim=embed_dim,
                                           padding_idx=0)
        
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=64,
                            batch_first=True)
        self.lstm_dropout = nn.Dropout(p=dropout_rate)
        
        # ==========================================
        # 2. 静态画像提取器 (处理稠密特征)
        # ==========================================
        self.portrait_mlp = nn.Sequential(
            nn.BatchNorm1d(portrait_dim),
            nn.Linear(portrait_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout_rate)
        )

        # ==========================================
        # 3. 特征融合与状态生成器
        # ==========================================
        self.item_feat_bn = nn.BatchNorm1d(slate_size * item_feat_dim)
        # 融合维度: 64(序列隐藏状态) + 64(画像特征) + 360(9个物品x40维属性)
        combined_dim = 64 + 64 + (slate_size * item_feat_dim)

        # 将融合特征降维并提取为 256 维的环境观测状态 (State)
        self.obs_layer = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ELU(),
            nn.Dropout(p=dropout_rate)
        )

        # ==========================================
        # 4. 奖励预测器
        # ==========================================
        # 根据当前状态，预测这 9 个物品被点击的独立概率 (0~1 之间)
        self.reward_layer = nn.Linear(256, slate_size)


    def forward(self, user_seq, user_portrait, item_feats):
        """
        前向传播：融合用户过去行为、静态画像与当前曝光物品，输出预测概率与环境状态
        user_seq: 用户过去点击的物品序列，[batch_size, seq_len]
        user_portrait: 用户静态画像，[batch_size, portrait_dim]
        item_feats: 曝光的物品embedding，[batch_size, slate_seq_len, embed_dim]
        """
        # --- 步骤 1: 提取动态兴趣 ---
        seq_emb = self.item_embedding(user_seq)  # [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        _, (h_n, _) = self.lstm(seq_emb)         # h_n 形状: [1, batch_size, 64]
        user_interest = h_n.squeeze(0)           # 降维后: [batch_size, 64]
        user_interest = self.lstm_dropout(user_interest)

        # --- 步骤 2: 提取静态画像 ---
        portrait_rep = self.portrait_mlp(user_portrait) # [batch_size, portrait_dim] -> [batch_size, 64]

        # --- 步骤 3: 处理当前曝光物品特征 ---
        flat_item_feats = item_feats.view(item_feats.size(0), -1) # [batch_size, 9, 40] -> [batch_size, 360]
        flat_item_feats = self.item_feat_bn(flat_item_feats)

        # --- 步骤 4: 特征拼接与环境状态生成 ---
        all_features = torch.cat([user_interest, portrait_rep, flat_item_feats], dim=-1) # [batch_size, 488]
        obs = self.obs_layer(all_features)      # [batch_size, 256] -> RL Agent 的观测 State

        # --- 步骤 5: 奖励预测 ---
        logits = self.reward_layer(obs)    # [batch_size, 9] -> 这批物品各自的点击概率
        click_probs = torch.sigmoid(logits)

        return click_probs, obs, logits