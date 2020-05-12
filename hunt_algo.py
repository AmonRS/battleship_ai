import random
import utils

hit_positions = []


def make_move(board):
    print("make move ()")
    print('hit_positions: ', hit_positions)
    size = len(board)
    # make random move if the stack is empty
    if not hit_positions:
        while True:
            x = random.randint(0, size - 1)
            y = random.randint(0, size - 1)

            if board[x][y] in ['~']:
                print('random move: ', x, y, '-----', board[x][y])
                board[x][y] = 'X'
                # dont change the hit flag even if the guessed position is not a ship
                break
                
            if (check_if_hit(board,x,y)):
                print ('random move: ', x, y, '-----', board[x][y])
                board[x][y] = 'X'
                hit_positions.append([x,y])
                break
    else:
        guess_along_hit(board)


def guess_along_hit(board):
    x = hit_positions[-1][0]
    y = hit_positions[-1][1]
    print("guess along hit () - ",x,' - ',y)
    utils.display_board(board)

    # start from top and move counterclockwise
    # also check if the coordinates are valid
    if(check_valid_coordinates(board,x,y-1) and board[x][y-1]!='X'):
        print('move after hit: ', x,y-1,' -----------', board[x][y-1])
        board[x][y-1] = 'X'
        if(check_if_hit(board, x, y-1)):
            hit_positions.append([x, y-1])
        return

    if(check_valid_coordinates(board,x-1,y) and board[x-1][y]!='X'):
        print('move after hit: ', x-1,y,' -----------', board[x-1][y])
        board[x-1][y] = 'X'
        if(check_if_hit(board, x-1, y)):
            hit_positions.append([x-1, y])
        return

    if(check_valid_coordinates(board,x,y+1) and board[x][y+1]!='X'):
        print('move after hit: ', x,y+1,' -----------', board[x][y+1])
        board[x][y+1] = 'X'
        if(check_if_hit(board, x, y+1)):
            hit_positions.append([x, y+1])
        return

    if(check_valid_coordinates(board,x+1,y) and board[x+1][y]!='X'):
        print('move after hit: ', x+1,y,' -----------', board[x+1][y])
        board[x+1][y] = 'X'
        if(check_if_hit(board, x+1, y)):
            hit_positions.append([x+1, y])
        return

    # if there is no hit, pop from stack
    hit_positions.pop
        
    


def check_valid_coordinates(board,x,y):
    if(x >= 0 & x <= len(board) & y >= 0 & y <= len(board)):
        return True
    return False


def check_if_hit(board, x, y):
    if board[x][y] in ['1', '2', '3', '4', '5']:
        return True
    return False






# copied this code from the file "random_algo.py"
def play(board):
    moves = 0

    while True:
        make_move(board)
        moves += 1
        print('moves: ',moves)
        # utils.display_board(board)

        if utils.check_if_won(board):
            break

    # print('random play done')
    utils.display_board(board)
    return moves