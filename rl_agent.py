import torch
import torch.optim as optim
from collections import deque
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy

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
    

class Actor(nn.Module):
    def __init__(
            self, 
            state_dim=cfg.state_dim, 
            slate_size=cfg.slate_size, 
            embed_dim=cfg.embed_dim
        ):
        super().__init__()
        self.slate_size = slate_size
        self.embed_dim = embed_dim

        # 输出动作总维数 slate_size * embed_dim
        self.action_flat_dim = slate_size * embed_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.action_flat_dim),
            # 使用Tanh将输出压缩到[-1.0, 1.0]
            nn.Tanh()
        )

    def forward(self, state):
        """
        输入：state [batch_size, state_dim(256)]
        输出：action [batch_size, slate_size(9), item_emb_dim(32)]
        """
        flat_action = self.net(state)
        # 构建动作矩阵，便于做KNN检索
        action_matrix = flat_action.view(-1, self.slate_size, self.embed_dim)
        return action_matrix
    

class Critic(nn.Module):
    def __init__(
            self, 
            state_dim=cfg.state_dim, 
            slate_size=cfg.slate_size, 
            embed_dim=cfg.embed_dim
        ):
        super().__init__()
        self.action_flat_dim = slate_size * embed_dim

        combined_dim = state_dim + self.action_flat_dim

        self.net = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            # 最后一层输出一个标量Q值
            # 因为推荐系统的Reward都是正数且肯恶搞较大，所以不加任何激活函数
            nn.Linear(512, 1)
        )

    def forward(self, state, action):
        """
        输入：state [batch_size, 256], action [batch_size, slate_size(9), item_emb_dim(32)]
        输出：Q-value [batch_size, 1]
        """
        flat_action = action.view(action.size(0), -1)   # [batch_size, 288]

        inputs = torch.cat([state, flat_action], dim=1) # [batch_size, 544]

        q_value = self.net(inputs)
        return q_value
    

class DDPGAgent:
    def __init__(
            self, 
            state_dim=cfg.state_dim, 
            slate_size=cfg.slate_size, 
            embed_dim=cfg.embed_dim, 
            device=cfg.device,
            gamma=cfg.gamma,
            tau=cfg.tau,
            expl_noise=cfg.expl_noise
        ):
        self.device = device
        self.slate_size = slate_size
        self.embed_dim = embed_dim

        self.actor = Actor(
            state_dim=state_dim,
            slate_size=slate_size, 
            embed_dim=embed_dim
        )

        self.critic = Critic(
            state_dim=state_dim,
            slate_size=slate_size,
            embed_dim=embed_dim
        )

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-4)

        self.gamma = gamma
        self.tau = tau
        self.expl_noise = expl_noise

    def select_action(self, state):
        """
        根据当前状态，返回一个带有噪声的动作，用于与环境交互
        """
        # numpy (CPU) -> Tensor (GPU)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).cpu().numpy()

        noise = np.random.normal(0, self.expl_noise, size=action.shape)
        action = np.clip(action + noise, -1.0, 1.0)

        return action
    
    def update(self, replay_buffer: ReplayBuffer, batch_size=cfg.batch_size):
        """
        DDPG 梯度下降逻辑
        """
        # 采样一批样本
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size=cfg.batch_size)

        # 更新 Critic
        with torch.no_grad():
            # Q = reward + gamma * next_Q * (1 - done)
            next_action = self.target_actor(next_states)
            target_q_next = self.target_critic(next_states, next_action)

        

    

if __name__ == "__main__":
    rb = ReplayBuffer()
    batch_size = 2
    for i in range(batch_size):
        state = [float(i + 1)] * 3
        action = [float(i + 2)] * 2
        reward = 10.0
        next_state = [float(i + 3)] * 3
        done = 0.0
        rb.push(state, action, reward, next_state, done)

    rb.sample(batch_size)


