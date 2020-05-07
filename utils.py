
import pprint
import random





# SHIPS

ships = [
    [1,'destroyer',2],
    [2,'submarine',3],
    [3,'cruiser',3],
    [4,'battleship',4],
    [5,'carrier',5]
]





# BOARD

def get_board(type='empty', size=10):
    board = []
    for i in range(size):
        board.append( ['~']*size )

    if type == 'empty':
        pass
    elif type == 'random_ships':
        for ship in ships:
            place_ship(board, ship)

    return board

def place_ship(board, ship):
    # place ship randomly onto the board
    size = len(board)
    while True:
        # align = random.choice('ver', 'hor')
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)

        blocked = 0
        for i in range(ship[2]):
            if y+i >= size:
                blocked = 1
            elif board[x][y+i] != '~':
                blocked = 1

        if blocked:
            continue    # try again
        else:
            for i in range(ship[2]):
                board[x][y+i] = str(ship[0])
            # ship placed successfully
            break


def display_board(board):
    pprint.pprint(board)

# board:
#   ~ = water/ocean
#   1-5 = one of the ships





# GAME

def check_if_won(board):
    # check if all the ships have been killed x_x
    pass