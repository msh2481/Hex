from time import sleep
import sys
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
from copy import deepcopy
from random import random, choice

class Turn:
    i: int
    j: int
 
    def __init__(self, i, j):
        self.i = i
        self.j = j
        # self.player = player
 
    def __str__(self):
        ans = str(self.i) + ' ' + str(self.j)
        return ans
 
 
class Board:
    n: int
    m: int
    player: int
    __dsu: list
    __size: list
    __history: list
    __pole: list
    __checkpoints: list

    def num(self, x, y):
        return self.m * x + y
 
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.player = 1
        self.__dsu = [i for i in range(n * m + 4)]
        self.__size = [1] * (n * m + 4)
        self.__pole = [0] * (n * m + 4)
        self.__history = list()
        self.__up = n * m
        self.__down = n * m + 1
        self.__left = n * m + 2
        self.__right = n * m + 3
        self.__pole[self.__up] = self.__pole[self.__down] = 1
        self.__pole[self.__left] = self.__pole[self.__right] = -1
        self.__checkpoints = [0]
 
    def parent(self, x):
        if x != self.__dsu[x]:
            return self.parent(self.__dsu[x])
        return x
 
    def save(self, array, idx):
        self.__history.append((array, idx, array[idx]))

    def merge(self, a, b):
        a = self.parent(a)
        b = self.parent(b)
        if a == b:
            return
        if self.__size[a] > self.__size[b]:
            a, b = b, a
        self.save(self.__dsu, a)
        self.save(self.__size, b)
        self.__dsu[a] = b
        self.__size[b] += self.__size[a]
 
    def rollback(self):
        assert self.__checkpoints
        need = self.__checkpoints.pop()
        while len(self.__history) > need:
            array, idx, old = self.__history.pop()
            array[idx] = old
        self.player *= -1
 
    def neighbours_list(self, x, y):
        ans = []
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, -1)]
        for dx, dy in moves:
            x2 = x + dx
            y2 = y + dy
            if 0 <= x2 < self.n and 0 <= y2 < self.m:
                ans.append(self.num(x2, y2))
        if x == 0:
            ans.append(self.__up)
        if x == self.n - 1:
            ans.append(self.__down)
        if y == 0:
            ans.append(self.__left)
        if y == self.m - 1:
            ans.append(self.__right)
        return ans
 
    def put(self, turn):
     #   print("Player", self.player)
        number = self.num(turn.i, turn.j)
        self.__checkpoints.append(len(self.__history))
        self.save(self.__pole, number)
        self.__pole[number] = self.player
        for neighbour in self.neighbours_list(turn.i, turn.j):
            if self.__pole[neighbour] == self.player:
                self.merge(neighbour, number)
        self.player *= -1
 
    def winner(self):
        if self.parent(self.__left) == self.parent(self.__right):
            return -1
        if self.parent(self.__up) == self.parent(self.__down):
            return 1
        return 0
    
    def moves(self):
        moves = []
        for i in range(self.n):
            for j in range(self.m):
                if self.__pole[self.num(i, j)] == 0:
                    moves.append(Turn(i, j))
        return moves
    
    def to_tensor(self):
        return torch.tensor(self.__pole[: self.n * self.m], dtype=torch.float)
    
    def debug(self):
        print('udlr', *[self.__dsu[i] for i in [self.__up, self.__down, self.__left, self.__right]])
        return self.__str__(debug=True)

    def __str__(self, debug=False):
        s = ''
        def what(i, j):
            if not debug:
                return '#.+'[self.__pole[self.num(i, j)] + 1]
            else:
                return str(self.__dsu[self.num(i, j)]%10)

        for i in range(self.n + 1):
            row = ''
            if i % 2 == 0:
                row += '\\_/ ' * (i // 2)
                for j in range(self.m - (i == self.n)):
                    row += '\\_/'
                    row += what(i, j) if i < self.n else ' '
                row += '\\'
            else:
                row += '/ \\_' * (i // 2 + (i >= 1))
                for j in range(self.m - (i == self.n)):
                    row += '/'
                    row += what(i, j) if i < self.n else ' '
                    row += '\\_'
            row = ' ' * ((i + (i == 0)) * 2) + row[((i + (i == 0)) * 2):]
            num = str(i) if i < self.n else ' '
            num += ' ' * (3 - len(num))
            s += num + row + '\n'
        first_line = "@"
        counter = 0
        while True:
            c = s[len(first_line)]
            if c == '\n':
                break
            if c in " \\_/":
                first_line += ' '
            else:
                first_line += str(counter)
                counter += 1
        result = first_line + '\n\n' + s
        result = result.replace('/', ' ')
        result = result.replace('\\', ' ')
        result = result.replace('_', ' ')
        return result

def get_params(model):
    return torch.cat(tuple(e.detach().flatten() for e in model.parameters()), dim = 0)
def eval_unity_root(poly, arg):
    num = np.exp(1j * arg)
    return np.polyval(poly, num)
def complex_hash(model, n):
    params = get_params(model)
    return np.abs(eval_unity_root(params, np.linspace(0, 2 * np.pi, num = n, endpoint = False)))

n = m = 5

class Player:
    history = []
    random_rate = 0.1
    discount_rate = 0.9
    learning = True
    success_story = []
    model = nn.Sequential(
        nn.Linear(n * m, 8), nn.ReLU(),
        nn.Linear(8, 4), nn.ReLU(),
        nn.Linear(4, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    def smart_select(self, board):
        def estimate(move):
            board.put(move)
            result = float(board.winner() == 1) if board.winner() else torch.sigmoid(self.model(board.to_tensor()))
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
        mse = 0
        for i in range(len(self.history) - 1):
            w = self.discount_rate ** (len(self.history) - 1 - i)
            mse += w * lf(self.model(self.history[i].to_tensor()), self.model(self.history[-1].to_tensor()))
        mse.backward()
        self.opt.step()
        self.success_story.append(complex_hash(self.model, 2))

    def remember_turn(self, board):
        if not self.history or self.history[-1] != board:
            self.history.append(board)
        else:
            # self play, avoid duplicating turns in history
            pass
    
    def make_move(self, board):
        self.remember_turn(board)
        self.study_last_turn()
        copy = deepcopy(board)
        if random() < self.random_rate:
            copy.put(self.rand_select(copy))
            self.history.clear()
            self.remember_turn(copy)
        else:
            copy.put(self.smart_select(copy))
            self.remember_turn(copy)
            self.study_last_turn()
        return copy
    
    def plot_success_story(self, f=None):
        print(f'story len is {len(self.success_story)}')
        if not self.success_story:
            return
        arr = np.array(self.success_story)
        plt.cla()
        plt.clf()
        plt.plot(arr[:, 0], arr[:, 1], 'o', ms=2)
        if not f:
            plt.show()
        else:
            plt.savefig(f)
            file = open(f[:-3] + 'txt', 'w')
            for e in self.success_story:
                print(e, file=file)
            file.close()

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

def contest(first, second):
    board = Board(n, m)
    while not board.winner():
        player = first if board.player == 1 else second
        temp = player.make_move(board)
        if temp:
            board = temp
        else:
            board.rollback()
            board.rollback()
    return board.winner()

def tournament(first, second, n = 1000):
    w = 0
    for game in range(n):
        if game % 2 == 0:
            w += contest(first, second) == 1
        else:
            w += contest(second, first) == -1
    return w / n

def training_camp(model, n = 10):
    old = deepcopy(model)
    model.learning = True
    for game in range(n):
        contest(model, model)
    new = deepcopy(model)
    old.learning = False
    new.learning = False
    # boost = tournament(new, old, n = 100)
    # print(f'boosted by {boost}', flush=True)
    return model

newbie = Player()
player = Player()
train = 10

for i in range(10**9):
    print(f'i = {i}, train = {train}, win = {tournament(player, newbie, train)}', flush=True)
    torch.save(player.model.state_dict(), f'zoo/{i}.pt')
    player.plot_success_story(f'plt/{i}.pdf')
    player = training_camp(player, train)
    train = int(1.1 * train)