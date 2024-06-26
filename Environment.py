"""
1. 要求能够自定义地图的大小
2. 起点和终点分别为左上角和右下角
3. 随机生成 20% 左右的坑
4. 到达终点给100奖励，掉坑或尝试走出边界给-100奖励

0 ,... , m-1
. ,... , .
. ,... , .
. ,... , .
(n-1)*m ,... , n*m-1

action: 1 2 3 4 上下左右
"""

# TODO 能不能加概率事件

import numpy as np
import pickle
import os
import json


class CliffEnv:
    def __init__(
        self,
        n=10,
        m=20,
        hole_rate=0.2,
        success_reward=100,
        failure_reward=-100,
        maximum_steps=100,
        load_path=None,
    ) -> None:
        self.n = n
        self.m = m
        self.hole_rate = hole_rate
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.maximum_steps = maximum_steps
        self.load_path = load_path

        self.reset()
        attr_dict = self.__dict__.copy()
        attr_dict.pop("holes")
        print("config:")
        print(json.dumps(attr_dict, indent=4))
        self.print_grid()

    def reset(self):
        self.pos = 0
        self.steps = 0
        self.done = False

        if not self.load_path:
            self.holes = np.random.choice(
                np.arange(1, self.n * self.m - 1),
                int((self.n * self.m - 2) * self.hole_rate),
                replace=False,
            )
        else:
            self.load_config()

        return self.pos

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
        self.steps += 1
        if self.steps >= self.maximum_steps:
            self.done = True
            return (self.pos, self.failure_reward, self.done)
        reward = 0
        x, y = self.pos // self.m, self.pos % self.m
        if action == 0:
            if x == 0:
                reward = self.failure_reward
                self.done = True
                return (self.pos, reward, self.done)
            self.pos -= self.m
        elif action == 1:
            if x == self.n - 1:
                reward = self.failure_reward
                self.done = True
                return (self.pos, reward, self.done)
            self.pos += self.m
        elif action == 2:
            if y == 0:
                reward = self.failure_reward
                self.done = True
                return (self.pos, reward, self.done)
            self.pos -= 1
        elif action == 3:
            if y == self.m - 1:
                reward = self.failure_reward
                self.done = True
                return (self.pos, reward, self.done)
            self.pos += 1
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

    def render(self):
        # TODO 显示画面
        pass

    def load_config(self):
        # 读取地图
        self.holes = np.load(os.path.join(self.load_path, "grid.npy"))
        # 读取参数
        with open(os.path.join(self.load_path, "config.pkl"), "rb") as f:
            config = pickle.load(f)
            self.n = config["n"]
            self.m = config["m"]
            self.hole_rate = config["hole_rate"]
            self.success_reward = config["success_reward"]
            self.failure_reward = config["failure_reward"]

    def save_config(self):
        self.save_dir = os.path.join(
            os.getcwd(),
            "map",
            str(self.n) + "x" + str(self.m),
            str(self.hole_rate),
        )  # 当前文件夹下的datas文件夹
        # 创建文件夹
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 保存地图
        np.save(os.path.join(self.save_dir, "grid.npy"), self.holes)
        with open(os.path.join(self.save_dir, "grid.txt"), "w") as f:
            for i in range(self.n):
                for j in range(self.m):
                    if i * self.m + j == self.pos:
                        f.write("A")
                    elif i * self.m + j in self.holes:
                        f.write("X")
                    elif i * self.m + j == self.n * self.m - 1:
                        f.write("B")
                    else:
                        f.write(".")
                f.write("\n")
        # 保存Env参数
        with open(os.path.join(self.save_dir, "config.pkl"), "wb") as f:
            pickle.dump(
                {
                    "n": self.n,
                    "m": self.m,
                    "hole_rate": self.hole_rate,
                    "success_reward": self.success_reward,
                    "failure_reward": self.failure_reward,
                },
                f,
            )


if __name__ == "__main__":
    env = CliffEnv(n=2, m=2, hole_rate=0.5)
    env.save_config()
