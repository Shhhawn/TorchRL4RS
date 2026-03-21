import os
from dataclasses import dataclass
import torch

@dataclass
class RL4RSConfig:
    # ===================================
    # 目录
    # ===================================
    data_dir: str = "../data/rl4rs_benchmark_materials/raw_data"
    train_data_file: str = "small_rl4rs_dataset_a_rl.csv"
    item_info_file: str = "item_info.csv"

    output_dir: str = "output"
    checkpoints_dir = "checkpoints"


    # ===================================
    # 数据处理与环境设置
    # ===================================
    # 用户历史行为序列阶段的最大长度
    max_seq_len: int = 50
    # Slate 推荐列表的长度（每次曝光 9 个商品）
    slate_size: int = 9
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ===================================
    # 模拟器网络超参数
    # ===================================
    sim_batch_size: int = 256
    sim_embed_dim: int = 32
    sim_learning_rate: float = 1e-3
    sim_epoch: int = 100
    sim_train_ratio: float = 0.8


    # ===================================
    # 数据集维度
    # ===================================
    # 物品个数
    item_vocab_size: int = 400000
    # 曝光列表长度
    slate_size: int = 9
    # 用户画像维度
    portrait_dim: int = 42
    # 物品特征维度
    item_feat_dim: int = 40
    # 历史序列长度
    max_seq_len: int = 50

    # embedding 维度（用于处理item_id和seq_item_id）
    embed_dim: int = 32

    def __post_init__(self):
        """这个方法会在对象初始化后自动执行"""
        os.makedirs(self.output_dir, exist_ok=True)

        self.train_path = os.path.join(self.data_dir, self.train_data_file)
        self.item_path = os.path.join(self.data_dir, self.item_info_file)
        self.checkpoints_path = os.path.join(self.output_dir, self.checkpoints_dir)

cfg = RL4RSConfig()
