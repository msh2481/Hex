from board import Turn, Board
from player import Bot, Human
from routines import tournament, training_camp 
from utils import git_save
from torch import save, load
from stats import player_stats

n = 5

newbie = Bot(n, n)
player = Bot(n, n)
train = 10
print('newbie', player_stats(newbie))

for i in range(10**9):
    git_save()
    print(f'i: {i}, train: {train}, win: {tournament(n, player, newbie, train)}', flush=True)
    print(player_stats(player))
    save(player.model.state_dict(), f'zoo/{i}.pt')
    player.plot_success_story(f'plt/{i}.pdf')
    player = training_camp(n, player, train)
    train = int(1.1 * train)
