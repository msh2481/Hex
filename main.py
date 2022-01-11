from board import Turn, Board
from player import Bot, Human
from routines import tournament, training_camp 
from utils import git_save
from torch import save, load
from stats import player_stats
from matplotlib import pyplot as plt
import wandb
wandb.init(project="hex-ai")

n = 5

newbie = Bot(n, n)
player = Bot(n, n)
train = 1
print('newbie', player_stats(newbie))

for i in range(10**9):
    # git_save()
    player = training_camp(n, player, train)
    train = int(1.1 * train) + 1
    print(player_stats(player))
    save(player.model.state_dict(), f'zoo/{i}.pt')
    player.plot_success_story()
    wandb.log({
        'train_steps': train,
        'win_rate': tournament(n, player, newbie, train),
        'chart': plt
    })
