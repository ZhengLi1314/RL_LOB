import shift
from RL_env import SHIFT_env as Env
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, DataParallelNet, Net
from tianshou.utils.net.discrete import Actor, Critic

action_dim = 3
state_dim = 4
device = 'cpu'
lr = 0.0003
buffer_size = 20000
epoch = 10,
step_per_epoch = 50000
repeat_per_collect = 10
test_num = 100
batch_size = 64

# symbol = 'GS'
# start = '2015-02-01'
# end = '2017-01-03'
# n_share = 100
seed = 13

trader = shift.Trader("test002")
trader.disconnect()
trader.connect("initiator.cfg", "password")
trader.subAllOrderBook()

env = Env(trader = trader,
          t = 1,
          nTimeStep=10,
          ODBK_range=5,
          symbol='CSCO',
          target_price=100)


np.random.seed(seed)
torch.manual_seed(seed)
train_envs.seed(seed)
test_envs.seed(seed)

net = Net(state_dim, hidden_sizes=(64, 64))
actor = Actor(net, action_dim, device=device).to(device)
critic = Critic(net, device=device).to(device)
actor_critic = ActorCritic(actor, critic)
optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)
dist = torch.distributions.Normal

policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist)

train_collector = Collector(
    policy, env, VectorReplayBuffer(buffer_size, 1)
)
test_collector = Collector(policy, env)

result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        epoch,
        step_per_epoch,
        repeat_per_collect,
        test_num,
        batch_size
    )

env.save_to_csv()
