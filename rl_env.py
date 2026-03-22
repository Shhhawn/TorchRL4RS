import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import gym
from gym import spaces

from nets import LSTMSlateSimulator
from config import cfg


class ItemFeatureManager:
    """根据物品ID查找物品特征向量"""
    def __init__(self, item_info_path, feat_dim=40, search_emb_dim=32, device=cfg.device):
        self.device = device
        self.feat_dim = feat_dim
        self.search_emb_dim = search_emb_dim

        df = pd.read_csv(item_info_path, sep=" ")
        max_id = df["item_id"].max()

        # 物品特征表
        self.item_feats = torch.zeros((max_id + 1, feat_dim), dtype=torch.float32, device=self.device)
        # 物品embedding表
        self.search_embs = torch.zeros((max_id + 1, search_emb_dim), dtype=torch.float32, device=self.device)
        # 物品价格表
        self.item_prices = torch.zeros(max_id + 1, dtype=torch.float32, device=self.device)

        for _, row in df.iterrows():
            item_id = int(row["item_id"])
            item_vec_list = [float(x) for x in str(row["item_vec"]).split(',')]
            self.item_feats[item_id] = torch.tensor(item_vec_list[-feat_dim:], dtype=torch.float32, device=self.device)
            self.search_embs[item_id] = torch.tensor(item_vec_list[-search_emb_dim:], dtype=torch.float32, device=self.device)
            self.item_prices[item_id] = float(row["price"])

        # 对所有物品embedding进行L2归一化，方便在与Agent输出的动作比较时进行点积计算相似度
        # 同时防止模长对相似度产生影响
        self.search_embs = F.normalize(self.search_embs, p=2, dim=1)
        print(f"词典构建完成！最大商品 ID: {max_id}")


    def get_feats(self, item_ids):
        return self.item_feats[item_ids]
    
    def get_prices(self, item_ids):
        return self.item_prices[item_ids]
    

class RL4RSEnv(gym.Env):
    def __init__(self, simulator_weights_path, item_info_path, user_data_path, device=cfg.device, max_steps=10):
        super().__init__()
        self.device = device
        self.slate_size = cfg.slate_size
        self.max_steps = max_steps

        # 动作空间定义：如果将动作设计成从所有物品中选出一个，重复slate_size次，agent网络最后一层就要输出每个物品的概率，
        # 最后一层会有几十万（400000）个输出节点，并要重复slate_size（9）次，即可能有400000^9种action的可能性，随时OOM；
        # 因此改成返回一个[slate_size, embed_dim]大小的矩阵，一行为一个Proto-Action（原型动作），即agent认为的理想物品，
        # 使用这个理想物品去物品库中用余弦相似度找最接近的物品，代表agent的推荐
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(cfg.slate_size, cfg.embed_dim),
            dtype=np.float32
        )

        # 状态空间定义
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32)


        # 加载模拟用户
        self.simulator = LSTMSlateSimulator(
            item_vocab_size=cfg.item_vocab_size,
            embed_dim=cfg.sim_embed_dim,
            portrait_dim=cfg.portrait_dim,
            item_feat_dim=cfg.item_feat_dim,
            slate_size=cfg.slate_size,
        ).to(self.device)

        # 加载预训练权重
        self.simulator.load_state_dict(torch.load(simulator_weights_path, map_location=self.device))
        self.simulator.eval()
        for param in self.simulator.parameters():
            param.requires_grad = False # 冻结模拟器


        # 初始化物品特征
        self.item_manager = ItemFeatureManager(
            item_info_path=item_info_path,
            feat_dim=cfg.item_feat_dim,
            device=self.device
        )


        # 准备用户初始状态
        self.user_pool = pd.read_csv(user_data_path, sep="@")

        self.current_user_seq = None
        self.current_user_portraits = None
        self.current_step = 0

    def _get_real_item_from_proto_action(self, proto_actions):
        """
        用于根据agent输出的Proto action找出最相似的物品
        proto_actions: RL Agent输出的连续向量，[slate_size, embed_dim]
        return: 最接近proto_actions的真实商品ID列表，[slate_size]
        """
        all_item_embs = self.item_manager.search_embs    # [item_vocab_size, emb_dim]
        scores = torch.matmul(proto_actions, all_item_embs.T)   # [slate_size, item_vocab_size],代表每个Proto-Action对所有商品的相似度(余弦相似度*|proto_actions|)，因为search_embs被L2归一化了，都是单位向量，因此不受物品本身模长影响

        # 沿着item_vocab_size的维度找相似度最大的下标即为最相似的物品下标
        real_item_ids = torch.argmax(scores, dim=1)

        return real_item_ids


    def _parse_seq(self, str_val, pad_len):
        """解析工具"""
        if pd.isna(str_val) or str_val == "": return [0] * pad_len
        lst = [int(x) for x in str(str_val).split(',')]
        if len(lst) >= pad_len: return lst[-pad_len:]
        return [0] * (pad_len - len(lst)) + lst
    

    def reset(self):
        self.current_step = 0
        # 随机采样一个用户
        sample_row = self.user_pool.sample(n=1).iloc[0]

        # 获取用户历史点击序列
        seq_list = self._parse_seq(sample_row['user_seqfeature'], cfg.max_seq_len)
        self.current_user_seq = torch.tensor(seq_list, dtype=torch.long, device=self.device).unsqueeze(0)   # [1, seq_len]

        # 获取用户静态画像
        portrait_list = [float(x) for x in str(sample_row["user_protrait"]).split(',')]
        self.current_user_portraits = torch.tensor(portrait_list, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 生成初始化 action，即占位推荐序列
        dummy_item_feats = torch.zeros((1, self.slate_size, cfg.item_feat_dim), device=self.device)

        # 获取一个初始state
        with torch.no_grad():
            _, obs, _ = self.simulator(self.current_user_seq, self.current_user_portraits, dummy_item_feats)

        return obs.squeeze(0).cpu().numpy()
    

    def step(self, action):
        """推荐序列action(CPU): [slate_size, emb_dim]"""
        self.current_step += 1

        # numpy (CPU) -> Tensor (GPU)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)

        # 拿真实 ID [slate_size]
        real_items = self._get_real_item_from_proto_action(action_tensor)
        # 拿真实item_feats
        item_feats = self.item_manager.get_feats(real_items).unsqueeze(0)
        # 拿真实item价格
        prices = self.item_manager.get_prices(real_items)

        # 计算一次点击概率，模拟用户点击
        with torch.no_grad():
            click_probs, _, _ = self.simulator(
                user_seq=self.current_user_seq, 
                user_portrait=self.current_user_portraits, 
                item_feats=item_feats
            )

        # 计算本次推荐的reward，即期望收益
        reward = torch.sum(click_probs.squeeze(0) * prices).item()
        # 伯努利采样将概率变成用1和0表示的真实点击，click_prob为0.7表示有0.7的概率为1（点击）
        clicks = torch.bernoulli(click_probs.squeeze(0))
        # 取出模拟出的被点击的物品，用于更新user_seq
        clicked_items = real_items[clicks == 1.0]

        num_clicks = clicked_items.size(0)

        if num_clicks > 0:
            old_seq = self.current_user_seq.squeeze(0)

            # 将模拟出的被点击的物品添加到user_seq中，剔除超出max_seq_len的老物品
            new_seq = torch.cat([old_seq, clicked_items], dim=0)
            new_seq = new_seq[-cfg.max_seq_len:]

            self.current_user_seq = new_seq.unsqueeze(0)

        """
        这里拿到点击的序列后需要重新计算一次next_obs
        当用户进入app界面后，agent推荐了slate_size个物品，上面的代码在模拟用户可能会点击哪几个物品；
        当用户点击了推荐的物品，推荐的物品存入了user_seq后，才进入到next_state，但是因为在next_state中还没有进行推荐，
        因此先使用dummy的全零item_feats占位
        """
        with torch.no_grad():
            _, next_obs, _ = self.simulator(
                user_seq=self.current_user_seq,
                user_portrait=self.current_user_portraits,
                item_feats=torch.zeros((1, self.slate_size, cfg.item_feat_dim), device=self.device)
            )

        done = self.current_step >= self.max_steps

        return next_obs.squeeze(0).cpu().numpy(), reward, done, {}
    


if __name__ == "__main__":
    import time
    simulator_weights = f"{cfg.checkpoints_path}/simulator_lstm_best.pth" 
    item_info = cfg.item_path
    user_data = cfg.train_path
    
    try:
        env = RL4RSEnv(
            simulator_weights_path=simulator_weights,
            item_info_path=item_info,
            user_data_path=user_data,
            device=cfg.device
        )
        print("环境实例化成功！\n")
        
        print("="*40)
        print("测试 1: 环境重置 (Reset)")
        print("="*40)
        
        start_time = time.time()
        obs = env.reset()
        reset_time = time.time() - start_time
        
        print(f"-> 成功抽样初始用户！")
        print(f"-> 初始观测状态 (Obs) 形状: {obs.shape} (期望是 (256,))")
        print(f"-> Reset 耗时: {reset_time:.4f} 秒\n")
        
        print("="*40)
        print("测试 2: 交互循环 (Step)")
        print("="*40)
        
        total_reward = 0.0
        # 故意循环 15 次，测试 max_steps=10 的截断机制是否生效
        for i in range(15):
            # 用 Gym 自带的随机动作采样器采样
            # 会生成一个形状为 (9, 32) 且数值在 [-1.0, 1.0] 之间的随机张量
            random_action = env.action_space.sample() 
            
            step_start = time.time()
            # 见证奇迹的时刻：把动作喂给环境
            next_obs, reward, done, info = env.step(random_action)
            step_time = time.time() - step_start
            
            total_reward += reward
            
            print(f"Step {i+1:02d} | 耗时: {step_time:.4f}s | 获得奖励 (Reward): {reward:.4f} | Done: {done}")
            
            if done:
                print("-" * 40)
                print(f"交互结束 (Episode 终止)！")
                print(f"随机策略总收益 (Total Expected Reward): {total_reward:.4f}")
                print(f"-> 最终次态 (Next Obs) 形状: {next_obs.shape}")
                break
                
    except Exception as e:
        print(f"\n环境自检失败，报错信息如下：\n{e}")
        import traceback
        traceback.print_exc()
