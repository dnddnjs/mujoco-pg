import numpy as np
import torch
from hparams import HyperParams as hp
from utils import log_density


def get_gae(rewards, masks, values):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    tderror = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        running_tderror = rewards[t] + hp.gamma * previous_value * masks[t] - \
                    values.data[t]
        running_advants = running_tderror + hp.gamma * hp.lamda * \
                          running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, tderror, advants


def surrogate_loss(actor, advants, states, old_policy, actions):
    mu, std, logstd = actor(torch.Tensor(states))
    new_policy = log_density(torch.Tensor(actions), mu, std, logstd)
    advants = advants.unsqueeze(1)
    surrogate = advants * torch.exp(new_policy - old_policy)
    surrogate = surrogate.mean()
    return - surrogate


def train_critic(critic, states, returns, critic_optim):
    criterion = torch.nn.MSELoss()
    n = len(states)
    for i in range(3):
        batch_index = np.random.randint(0, n, size=hp.batch_size)
        batch_index = torch.LongTensor(batch_index)
        inputs = torch.Tensor(states)[batch_index]
        targets = returns.unsqueeze(1)[batch_index]

        values = critic(inputs)
        loss = criterion(values, targets)
        critic_optim.zero_grad()
        loss.backward()
        critic_optim.step()


def train_actor(actor, advants, states, old_policy, actions, actor_optim):
    loss = surrogate_loss(actor, advants, states, old_policy, actions)
    actor_optim.zero_grad()
    loss.backward()
    actor_optim.step()


def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])
    values = critic(torch.Tensor(states))

    returns, tderror, advants = get_gae(rewards, masks, values)
    mu, std, logstd = actor(torch.Tensor(states))
    old_policy = log_density(torch.Tensor(actions), mu, std, logstd)

    train_critic(critic, states, returns, critic_optim)
    train_actor(actor, advants, states, old_policy, actions, actor_optim)
    return returns


