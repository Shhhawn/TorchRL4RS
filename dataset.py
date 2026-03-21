import torch
from torch.utils.data import Dataset
import pandas as pd

from config import cfg

class RL4RSSimulatorDataset(Dataset):
    def __init__(self, data_file=cfg.train_path, max_seq_len=50):
        """
        读取并解析 RL4RS 的脱敏日志数据
        """
        print(f"正在加载数据：{data_file}")
        self.data = pd.read_csv(data_file, sep="@")
        self.max_seq_len = max_seq_len
        print(f"成功加载{len(self.data)}条会话记录")

    def __len__(self):
        return len(self.data)
    
    def _parse_seq(self, str_val, pad_len=None):
        """
        将逗号分隔的字符串解析成定长Tensor
        eg. "15,15,21,37,132" -> tensor([0, 0, 15, 15, 21, 37, 132])
        """
        if pd.isna(str_val) or str_val == "":
            lst = []
        else:
            lst = [int(x) for x in str(str_val).split(',')]

        if len(lst) >= pad_len:
            # 如果序列太长，保留最近的pad_len条交互记录
            lst = lst[-pad_len:]
        else:
            # 如果序列太短，用0补齐
            lst = [0] * (pad_len - len(lst)) + lst

        return torch.tensor(lst, dtype=torch.long)
    
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq_tensor = self._parse_seq(row['user_seqfeature'], self.max_seq_len)

        portrait_tensor = torch.tensor(
            [float(x) for x in str(row["user_protrait"]).split(',')],
            dtype = torch.float32
        )

        # 动作：曝光的9个物品ID
        action_ids = torch.tensor(
            [int(x) for x in str(row["exposed_items"]).split(',')],
            dtype = torch.long
        )

        # 曝光商品的40维特征
        item_feat_list = []
        for item in str(row["item_feature"]).split(';'):
            item_feat_list.append([float(x) for x in item.split(',')])
        item_feat_tensor = torch.tensor(item_feat_list, dtype=torch.float32)

        # 奖励：用户点击反馈
        labels = torch.tensor(
            [float(x) for x in str(row["user_feedback"]).split(',')],
            dtype = torch.float32
        )

        return {
            "user_seq": seq_tensor,
            "user_portrait": portrait_tensor,
            "action_ids": action_ids,
            "item_feats": item_feat_tensor,
            "labels": labels
        }


if __name__ == "__main__":
    data_file = cfg.train_path
    dataset = RL4RSSimulatorDataset(data_file, max_seq_len=8)
    test_str = "10,4,2,25"
    print(f"原始数据： {test_str}")
    print(f"处理后数据： {dataset._parse_int_tensor(test_str, pad_len=8)}")