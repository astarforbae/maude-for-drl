"""
1. 要求能够自定义地图的大小
2. 起点和终点分别为左上角和右下角
3. 随机生成 20% 左右的坑
4. 到达终点给100奖励，掉坑给-100奖励

0 ,... , m-1
. ,... , .
. ,... , .
. ,... , .
(n-1)*m ,... , n*m-1

action: 0 1 2 3 4 上下左右静止
"""

# TODO 能不能加概率事件

import numpy as np
import pickle
import datetime
import os
import json


class CliffEnv:
    def __init__(
        self,
        n=10,
        m=20,
        hole_rate=0.2,
        reward_decay=True,
        success_reward=100,
        failure_reward=-100,
        saved_config_path=None,
    ) -> None:
        self.n = n
        self.m = m
        self.hole_rate = hole_rate
        self.reward_decay = reward_decay
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.saved_config_path = saved_config_path

        self.save_dir = os.path.join(
            os.getcwd(),
            "data",
            str(self.n) + "x" + str(self.m),
            datetime.datetime.now().strftime("%m_%d_%H_%M_%S"),
        )  # 当前文件夹下的datas文件夹
        # 创建文件夹
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        print("dir: " + self.save_dir)
        self.reset()

    def reset(self):
        self.pos = 0
        self.done = False

        if not self.saved_config_path:
            self.holes = np.random.choice(
                np.arange(1, self.n * self.m),
                int((self.n * self.m - 2) * self.hole_rate),
                replace=False,
            )
        else:
            self.load_config()
        self.print_grid()
        attr_dict = self.__dict__.copy()
        attr_dict.pop("holes")
        print("config:")
        print(json.dumps(attr_dict, indent=4))

    def print_grid(self):
        for i in range(self.n):
            for j in range(self.m):
                if i * self.m + j == self.pos:
                    print("A", end=" ")
                elif i * self.m + j in self.holes:
                    print("X", end=" ")
                elif i * self.m + j == self.n * self.m - 1:
                    print("B", end=" ")
                else:
                    print(".", end=" ")
            print()

    def step(self, action):
        """
        return (pos, reward, done)
        """
        if action == 0:
            self.pos -= self.m
        elif action == 1:
            self.pos += self.m
        elif action == 2:
            self.pos -= 1
        elif action == 3:
            self.pos += 1
        reward = 0
        if self.pos in self.holes:
            reward = self.failure_reward
            self.done = True
        elif self.pos == self.n * self.m - 1:
            reward = self.success_reward
            self.done = True
        return (self.pos, reward, self.done)

    def available_actions(self):
        x, y = self.pos // self.m, self.pos % self.m
        actions = range(5)
        if x == 0:
            actions.remove(0)
        if x == self.n - 1:
            actions.remove(1)
        if y == 0:
            actions.remove(2)
        if y == self.m - 1:
            actions.remove(3)
        return actions

    def load_config(self):
        # 读取地图
        self.holes = np.load(os.path.join(self.saved_config_path, "grid.npy"))
        # 读取参数
        with open(os.path.join(self.saved_config_path, "config.pkl"), "rb") as f:
            config = pickle.load(f)
            self.n = config["n"]
            self.m = config["m"]
            self.hole_rate = config["hole_rate"]
            self.reward_decay = config["reward_decay"]
            self.success_reward = config["success_reward"]
            self.failure_reward = config["failure_reward"]

    def save_config(self):
        # 保存地图
        np.save(os.path.join(self.save_dir, "grid.npy"), self.holes)
        # 保存Env参数
        with open(os.path.join(self.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(
                {
                    "n": self.n,
                    "m": self.m,
                    "hole_rate": self.hole_rate,
                    "reward_decay": self.reward_decay,
                    "success_reward": self.success_reward,
                    "failure_reward": self.failure_reward,
                },
                f,
            )


# env = CliffEnv(saved_config_path="/home/ZhangXingYi/codes/CLIFF/data/10x20/base_config")
# env.save_config()
