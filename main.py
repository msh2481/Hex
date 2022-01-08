import torch
import numpy as np
from torch import nn
from copy import deepcopy
from board import Turn, Board
from player import Bot, Human
from routines import tournament, training_camp
from time import time 
from utils import git_save

n = 5

newbie = Player(n, n)
player = Player(n, n)
train = 10

for i in range(10**9):
    git_save()
    print(f'i: {i}, train: {train}, win: {tournament(player, newbie, train)}', flush=True)
    torch.save(player.model.state_dict(), f'zoo/{i}.pt')
    player.plot_success_story(f'plt/{i}.pdf')
    player = training_camp(player, train)
    train = int(1.1 * train)
