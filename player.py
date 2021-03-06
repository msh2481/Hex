from random import random, choice
from stats import complex_hash
from matplotlib import pyplot as plt
from utils import log
import torch
from torch import nn
import numpy as np
from copy import deepcopy
from board import Board 


best = 1

class Bot:
    def __init__(self, n, m):
        self.history = []
        self.random_rate = 0.1
        self.discount_rate = 0.5
        self.learning = True
        self.success_story = []
        self.model = nn.Sequential(
            nn.Linear(2 * n * m + 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-2, weight_decay=1e-8)

    def estimate_first(self, board):
        assert type(board) is Board
        global best
        if board.winner():
            inf = 1e3
            result = torch.tensor(inf if board.winner() == 1 else -inf, dtype=torch.float)
        else:
            # result = torch.sigmoid(self.model(board.to_tensor()))
            result = self.model(board.to_tensor())
        if 0 < min(result, 1 - result) < best:
            best = min(result, 1 - result)
            # print(board)
            # print('est = ', result, flush=True)
        return result

    def smart_select(self, board):
        assert type(board) is Board
        def estimate(move):
            board.put(move)
            assert type(board) is Board
            result = self.estimate_first(board)
            board.rollback()
            return result if board.player == 1 else 1 - result
        return max(board.moves(), key=estimate)
    
    def rand_select(self, board):
        return choice(board.moves())
    
    def study_last_turn(self):
        if not self.learning or len(self.history) < 2:
            return
        self.opt.zero_grad()
        def lf(x, y):
            return (x-y)**2

        # print('---------------', file=log)
        # for e in self.history:
        #     print(f'win: {self.estimate_first(e).item()}', file=log)
        #     print(e, file=log)
        # print(flush=True, file=log)

        mse = lf(self.estimate_first(self.history[-2]), self.estimate_first(self.history[-1]))
        # for i in range(len(self.history) - 1):
        #     w = self.discount_rate ** (len(self.history) - 1 - i)
        #     mse += w * lf(self.estimate_first(self.history[i]), self.estimate_first(self.history[-1]))
        mse.backward()
        self.opt.step()
        self.success_story.append(complex_hash(self.model, 2))

    def remember_turn(self, board):
        if not self.history or self.history[-1] != board:
            self.history.append(board)
        else:
            # self play, avoid duplicating turns in history
            pass
    
    def clear_history(self):
        self.history.clear()

    def make_move(self, board):
        self.remember_turn(board)
        self.study_last_turn()
        copy = deepcopy(board)
        if random() < self.random_rate:
            copy.put(self.rand_select(copy))
            self.clear_history()
            self.remember_turn(copy)
        else:
            copy.put(self.smart_select(copy))
            self.remember_turn(copy)
            self.study_last_turn()
        return copy
    
    def plot_success_story(self, f=None):
        # print(f'story len is {len(self.success_story)}')
        if not self.success_story:
            return
        arr = np.array(self.success_story)
        plt.cla()
        plt.clf()
        plt.plot(arr[:, 0], arr[:, 1], 'o-', ms=2)
        # if not f:
        #     plt.show()
        # else:
        #     plt.savefig(f)
        #     file = open(f[:-3] + 'txt', 'w')
        #     for e in self.success_story:
        #         print(e, file=file)
        #     file.close()

class Human:
    def make_move(self, board):
        assert not board.winner()
        print(board)
        try:
            i, j = map(int, input().split())
            copy = deepcopy(board)
            copy.put(Turn(i, j))
            return copy
        except:
            return None

    def clear_history():
        print('Starting new game')