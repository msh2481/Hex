from board import Board, Turn

positions = []

def add_position(name, turns, n=5):
	global positions
	board = Board(n, n)
	for i, j in turns:
		board.put(Turn(i, j))
	positions.append((name, board))

add_position('0.0, 1 turn', [
	(1, 0),
	(0, 1),
	(2, 0),
	(0, 2),
	(3, 0),
	(0, 3),
	(4, 0),
	(0, 4),
	(4, 4)
])


add_position('1.0, 1 turn', [
	(1, 0),
	(0, 1),
	(2, 0),
	(0, 2),
	(3, 0),
	(0, 3),
	(4, 0),
	(0, 4)
])

# print(positions[0][1])
# positions[0][1].put(Turn(0, 0))
# print(positions[0][1].winner())