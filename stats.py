import torch
import numpy as np

def get_params(model):
    return torch.cat(tuple(e.detach().flatten() for e in model.parameters()), dim = 0)
def eval_unity_root(poly, arg):
    num = np.exp(1j * arg)
    return np.polyval(poly, num)
def complex_hash(model, n):
    params = get_params(model)
    return np.abs(eval_unity_root(params, np.linspace(0, 2 * np.pi, num = n, endpoint = False)))

def player_stats(player):
    from positions import positions
    return {name: player.estimate_first(board) for name, board in positions}
