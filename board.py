import torch
import numpy as np

class HexBoard:
    def __init__(self, n, frame):
        self.from_list(n, frame, [])

    def inside(self, i, j):
        return i in range(0, self.s) and j in range(0, self.s)

    def neighbours_list(self, i, j):
        return [(i + di, j + dj) for di, dj in self.moves if self.inside(i + di, j + dj)]

    def bfs(self, start, color):
        from collections import deque
        q = deque([(i, j) for (i, j) in start if self.board[i][j] == color])
        used = np.zeros((self.s, self.s), dtype=bool)
        for i, j in q:
            used[i][j] = True
        while q:
            i, j = q.popleft()
            assert self.board[i][j] == color and used[i][j]
            for ni, nj in self.neighbours_list(i, j):
                if self.board[ni][nj] != color or used[ni][nj]:
                    continue
                used[ni][nj] = True
                q.append((ni, nj))
        return used

    def current_player(self):
        balance = self.board.sum()
        return 1 if balance == 0 else -1

    def get(self, i, j):
        assert self.inside(i + self.frame, j + self.frame), 'position out of bounds'
        return self.board[i + self.frame][j + self.frame]

    def put(self, i, j, c=None):
        c = c or self.current_player()
        assert not self.get(i, j), 'cell is not empty'
        self.board[i + self.frame][j + self.frame] = c

    def __str__(self, d=2, axes=True):
        from itertools import product
        tokens = {None: ' ', 0: '_', 1: '@', -1: 'X'}
        ivec, jvec = np.array([(d*3**0.5+1)//2, (d+1)//2], dtype=int), np.array([0, d//1], dtype=int)
        h, w = (ivec + jvec) * (self.s - 1) + 1
        res = [[tokens[None] for j in range(w)] for i in range(h)]
        for i, j in product(range(self.s), repeat=2):
            pos = i * ivec + j * jvec
            res[pos[0]][pos[1]] = tokens[self.board[i][j]]
        return '\n'.join([''.join(line) for line in res])

    def to_list(self):
        full = [(i, j, self.get(i, j)) for i, j in product(range(self.n), repeat=2)]
        return [x for x in full if x[2]]

    def from_list(self, n, frame, l):
        self.n = n
        self.frame = frame
        self.s = n + 2 * frame
        self.board = np.zeros((self.s, self.s), dtype=int)
        self.board[0:self.frame, :] += 1
        self.board[-self.frame:, :] += 1
        self.board[:, 0:self.frame] -= 1
        self.board[:, -self.frame:] -= 1
        self.moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, 1), (1, -1)]
        for i, j, c in l:
            self.put(i, j, c)
        self.up = [(self.frame, self.frame + j) for j in range(self.n)]
        self.down = [(self.frame + self.n - 1, self.frame + j) for j in range(self.n)]
        self.left = [(self.frame + i, self.frame) for i in range(self.n)]
        self.right = [(self.frame + i, self.frame + self.n - 1) for i in range(self.n)]

    def win(self):
        first = (self.bfs(self.up, 1) & self.bfs(self.down, 1)).sum() > 0
        second = (self.bfs(self.left, -1) & self.bfs(self.right, -1)).sum() > 0
        assert not (first and second)
        return int(first) - int(second)

    def to_tensors(self):
        return np.stack([
            self.board == 1,
            self.board == -1,
            self.bfs(self.up, 1),
            self.bfs(self.down, 1),
            self.bfs(self.left, -1),
            self.bfs(self.right, -1)
        ]).astype(np.float32)

def to_str(tensor):
    from copy import deepcopy
    a = deepcopy(tensor).astype(str)
    a[a == 'False'] = '_'
    a[a == 'True'] = '#'
    return a

import gym
from gym import spaces

class HexEnv(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self, n, frame):
        super(HexEnv, self).__init__()
        self.n = n
        self.frame = frame
        self.s = self.n + 2 * self.frame
        self.action_space = spaces.MultiDiscrete([self.n, self.n], dtype=np.int32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6, self.s, self.s), dtype=np.float32)
        self.board = HexBoard(self.n, self.frame)
    def step(self, action):
        print(type(action), action)
        assert type(action) is np.ndarray
        assert action.shape == (2,)
        assert action.dtype == np.int32
        done = False
        reward = -1
        info = {}
        if not self.board.get(*action):
            self.board.put(*action)
            w = self.board.win()
            done = w != 0
            reward = w
        return self.board.to_tensors(), reward, done, info
    def reset(self):
        self.board = HexBoard(self.n, self.frame)
        return self.board.to_tensors()
    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(self.bord)
    def close(self):
        pass
    def seed(self, seed):
        np.random.seed(seed)