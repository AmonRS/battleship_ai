
import pprint
import random





# SHIPS

ships = [
    [1, 'destroyer', 2],            # [ no. , name , size]
    [2, 'submarine', 3],            # no. is used to identify ships on the board 
    [3, 'cruiser', 3], 
    [4, 'battleship', 4], 
    [5, 'carrier', 5]
]





# BOARD

def get_board(type='empty', size=10):
    # empty board
    board = []
    for i in range(size):
        board.append( ['~']*size )      # '~' is the ocean/water :)

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
        # align = random.choice('ver', 'hor') @todo              # right now the ships are only placed horizontally
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)

        blocked = 0
        for i in range(ship[2]):
            if y+i >= size:
                blocked = 1
            elif board[x][y+i] != '~':
                blocked = 1

        if blocked:
            continue # try again
        else:
            for i in range(ship[2]):
                board[x][y+i] = str(ship[0])
            break # ship placed successfully


def display_board(board):
    pprint.pprint(board)

# board:
#   ~ = water/ocean
#   1-5 = one of the ships
#   X = ship got hit
#   0 = hit water





# GAME

def check_if_won(board):
    # check if all the ships have been killed x_x
    size = len(board)
    for i in range(size):
        for j in range(size):
            if board[i][j] in ['1','2','3','4','5']:
                return False
    
    return True # no more ships left


# feedback for the algorithms to know how many ships are sunk
def get_sunk_ships(board):
    ships_sunk = []

    size = len(board)
    for i in range(size):
        for j in range(size):
            if not board[i][j] in ['1']:
                ships_sunk.append(1)
            if not board[i][j] in ['2']:
                ships_sunk.append(2)
            if not board[i][j] in ['3']:
                ships_sunk.append(3)
            if not board[i][j] in ['4']:
                ships_sunk.append(4)
            if not board[i][j] in ['5']:
                ships_sunk.append(5)
    
    return ships_sunk