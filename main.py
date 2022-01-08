import torch
import numpy as np
from torch import nn
from copy import deepcopy
from board import Turn, Board
from player import Bot, Human
from routines import tournament, training_camp
from time import time 

def git_save():
    from os import system as cmd
    print(cmd('git switch autosave'))
    print(cmd('git add .'))
    print(cmd('git commit -m' + str(int(time()))))
    print(cmd('git push'))

# n = 5

# newbie = Player(n, n)
# player = Player(n, n)
# train = 10

# for i in range(10**9):
#     print(f'i: {i}, train: {train}, win: {tournament(player, newbie, train)}', flush=True)
#     torch.save(player.model.state_dict(), f'zoo/{i}.pt')
#     player.plot_success_story(f'plt/{i}.pdf')
#     player = training_camp(player, train)
#     train = int(1.1 * train)

git_save()