import torch

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
        a = self.__pole[: self.n * self.m]
        turns = sum([x != 0 for x in a])
        return torch.tensor([x == 1 for x in a] + [x == -1 for x in a] + [turns, turns % 2], dtype=torch.float)
    
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