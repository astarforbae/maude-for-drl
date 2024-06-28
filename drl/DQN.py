import random
import os
import json
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
from Environment import CliffEnv
import matplotlib.pyplot as plt


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class Qnet(torch.nn.Module):
    """只有一层隐藏层的Q网络"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        # self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        # x = F.relu(self.fc2(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DQN:
    """DQN算法"""

    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        learning_rate,
        gamma,
        epsilon,
        target_update,
        device,
    ):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action_deterministic(self, state):  # 确定性策略采取动作
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.q_net(state).argmax().item()
        return action

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = self.take_action_deterministic(state)
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float).to(
            self.device
        )
        actions = torch.tensor(transition_dict["actions"]).view(-1, 1).to(self.device)
        rewards = (
            torch.tensor(transition_dict["rewards"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        next_states = torch.tensor(
            transition_dict["next_states"], dtype=torch.float
        ).to(self.device)
        dones = (
            torch.tensor(transition_dict["dones"], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_next_run_id(path):
    if not os.path.exists(path):
        return 0
    else:
        return (
            max(
                [
                    int(x)
                    for x in os.listdir(path)
                    if os.path.isdir(os.path.join(path, x)) and x.isdigit()
                ]
                + [0]
            )
            + 1
        )


def print_action_table(agent, env, device):
    action_map = {
        0: "↑",
        1: "↓",
        2: "←",
        3: "→",
    }
    for i in range(env.n):
        for j in range(env.m):
            if i * env.m + j in env.holes:
                print("X", end="")
                continue
            action = (
                agent.q_net(torch.tensor([i, j], dtype=torch.float).to(device))
                .argmax()
                .item()
            )
            print(action_map[action], end="")
        print()


def print_action_table_to_file(agent, env, device, file_path):
    action_map = {
        0: "↑",
        1: "↓",
        2: "←",
        3: "→",
    }
    with open(file_path, "w") as f:
        for i in range(env.n):
            for j in range(env.m):
                if i * env.m + j in env.holes:
                    f.write("X")
                    continue
                action = (
                    agent.q_net(torch.tensor([i, j], dtype=torch.float).to(device))
                    .argmax()
                    .item()
                )
                f.write(action_map[action])
            f.write("\n")


def train(
    state_dim,
    action_dim,
    lr,
    num_episodes,
    hidden_dim,
    gamma,
    epsilon,
    target_update,
    buffer_size,
    minimal_size,
    batch_size,
    device,
    map_path,
):
    params = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "lr": lr,
        "num_episodes": num_episodes,
        "hidden_dim": hidden_dim,
        "gamma": gamma,
        "epsilon": epsilon,
        "target_update": target_update,
        "buffer_size": buffer_size,
        "minimal_size": minimal_size,
        "batch_size": batch_size,
        "device": device,
        "map_path": map_path,
    }
    env = CliffEnv(load_path=map_path)
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DQN(
        state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device
    )

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            cnt = 0
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            "states": b_s,
                            "actions": b_a,
                            "next_states": b_ns,
                            "rewards": b_r,
                            "dones": b_d,
                        }
                        agent.update(transition_dict)
                if reward == env.success_reward:
                    cnt += 1
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
            print_action_table(agent, env, device)
    # 保存模型参数
    save_path = os.path.join(map_path, str(get_next_run_id(map_path)))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(
        agent.q_net.state_dict(),
        os.path.join(save_path, "q_net.pth"),
    )

    # 保存参数
    with open(os.path.join(save_path, "parameters.json"), "w") as f:
        json.dump(params, f)
    # 画return图
    avg_return_list = np.convolve(return_list, np.ones(10) / 10, mode="valid")
    plt.plot(avg_return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.savefig(os.path.join(save_path, "return.png"))
    # 打印在每个位置上的动作
    print_action_table_to_file(
        agent, env, device, os.path.join(save_path, "actions.txt")
    )
    print(json.dumps(params, indent=4))
    print("Save to %s" % save_path)


def eval(model_path=None):
    assert model_path is not None and os.path.exists(model_path)
    with open(os.path.join(model_path, "parameters.json"), "r") as f:
        params = json.load(f)

    # 加载地图
    env = CliffEnv(load_path=params["map_path"])
    state_dim = params["state_dim"]
    action_dim = params["action_dim"]
    lr = params["lr"]
    hidden_dim = params["hidden_dim"]
    gamma = params["gamma"]
    epsilon = params["epsilon"]
    target_update = params["target_update"]

    # 加载模型
    device = torch.device(params["device"])
    agent = DQN(
        state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device
    )
    agent.q_net.load_state_dict(torch.load(os.path.join(model_path, "q_net.pth")))
    # TODO eval 要干什么
