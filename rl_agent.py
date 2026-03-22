import torch
from collections import deque
import random
import numpy as np

from config import cfg

class ReplayBuffer:
    """
    经验回访池：用于存储智能体与环境的交互记录
    """
    def __init__(self, capacity=cfg.replay_buffer_size, device=cfg.device):
        """
        capacity: 经验回访池的最大容量
        """
        self.buffer = deque(maxlen = capacity)
        self.device = device
    
    def push(self, state, action, reward, next_state, done):
        """
        将经验存入经验回放池
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        随机抽取一个batch的数据，并拼接、转换成GPU Tensor
        """
        batch = random.sample(self.buffer, batch_size)

        state, action, reward, next_state, done = zip(*batch)

        # [batch_size, state_dim]
        state_batch_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=self.device)
        # [batch_size, slate_size, item_emb_dim]
        action_batch_tensor = torch.tensor(np.array(action), dtype=torch.float32, device=self.device)
        # [batch_size, 1]
        reward_batch_tensor = torch.tensor(np.array(reward), dtype=torch.float32, device=self.device).unsqueeze(1)
        # [batch_size, state_dim]
        next_state_batch_tensor = torch.tensor(np.array(next_state), dtype=torch.float32, device=self.device)
        # [batch_size, 1]
        done_batch_tensor = torch.tensor(np.array(done), dtype=torch.float32, device=self.device).unsqueeze(1)

        return state_batch_tensor, action_batch_tensor, reward_batch_tensor, next_state_batch_tensor, done_batch_tensor
    

    def __len__(self):
        """
        返回当前有多少数据
        """
        return len(self.buffer)