import torch
from DQN import train
from DQN import set_seed

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = "cpu"
set_seed(0)

train(
    state_dim=2,
    action_dim=4,
    lr=1e-3,
    num_episodes=20000,
    hidden_dim=128,
    gamma=0.99,
    epsilon=0.01,
    target_update=10,
    buffer_size=100000,
    minimal_size=5000,
    batch_size=32,
    device=device,
    map_path="/home/ZhangXingYi/codes/CLIFF/map/3x3/0",
)
