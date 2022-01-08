from board import Board
from copy import deepcopy

def contest(size, first, second):
    board = Board(size, size)
    while not board.winner():
        player = first if board.player == 1 else second
        temp = player.make_move(board)
        if temp:
            board = temp
        else:
            board.rollback()
            board.rollback()
    return board.winner()

def tournament(size, first, second, n = 1000):
    w = 0
    for game in range(n):
        if game % 2 == 0:
            w += contest(size, first, second) == 1
        else:
            w += contest(size, second, first) == -1
    return w / n

def training_camp(size, model, n = 10):
    old = deepcopy(model)
    model.learning = True
    for game in range(n):
        contest(size, model, model)
    new = deepcopy(model)
    old.learning = False
    new.learning = False
    # boost = tournament(new, old, n = 100)
    # print(f'boosted by {boost}', flush=True)
    return model