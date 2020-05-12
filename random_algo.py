import random

import utils





def make_random_move(board):
    size = len(board)
    while True:
        x = random.randint(0, size-1)
        y = random.randint(0, size-1)

        if board[x][y] in ['~','1','2','3','4','5']:
            print('random move: ',x,y, '-----',board[x][y])
            board[x][y] = 'X'
            break



def play(board):
    # print('starting random play')
    moves = 0

    while True:
        make_random_move(board)
        moves += 1
        # utils.display_board(board)

        if utils.check_if_won(board):
            break

    # print('random play done')
    utils.display_board(board)
    return moves