import torch


def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action