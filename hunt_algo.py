import random
import utils

hit_positions = []


def make_move(board):
    print("make move ()")
    print('hit_positions: ', hit_positions)

    # make random move if the stack is empty
    if not hit_positions:
        make_random_move(board)
    else:
        guess_along_hit(board)


def make_random_move(board):
    size = len(board)

    while True:
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)

        if board[x][y] in ['~']:
            print('random move: ', x, y, '-----', board[x][y])
            board[x][y] = '0'
            # dont change the hit flag even if the guessed position is not a ship
            break
            
        if (check_if_hit(board,x,y)):
            print ('random move: ', x, y, '-----', board[x][y])
            board[x][y] = 'X'
            hit_positions.append([x,y])
            break


def guess_along_hit(board, hit_positions = hit_positions):
    x = hit_positions[-1][0]
    y = hit_positions[-1][1]
    print(x, y)


    # also check if the coordinates are valid
    if(check_valid_coordinates(board,x,y-1) and not board[x][y-1] in ['X','0']):
        print('move after hit: ', x,y-1,' -----------', board[x][y-1])
        
        if(check_if_hit(board, x, y-1)):
            hit_positions.append([x, y-1])
            board[x][y-1] = 'X'
            return
        board[x][y-1] = '0'
        return

    if(check_valid_coordinates(board,x-1,y) and not board[x-1][y] in ['X','0']):
        print('move after hit: ', x-1,y,' -----------', board[x-1][y])
        if(check_if_hit(board, x-1, y)):
            hit_positions.append([x-1, y])
            board[x-1][y] = 'X'
            return
        board[x-1][y] = '0'
        return

    if(check_valid_coordinates(board,x,y+1) and not board[x][y+1] in ['X','0']):
        print('move after hit: ', x,y+1,' -----------', board[x][y+1])
        if(check_if_hit(board, x, y+1)):
            hit_positions.append([x, y+1])
            board[x][y+1] = 'X'
            return
        board[x][y+1] = '0'
        return

    if(check_valid_coordinates(board,x+1,y) and not board[x+1][y] in ['X','0']):
        print('move after hit: ', x+1,y,' -----------', board[x+1][y])
        if(check_if_hit(board, x+1, y)):
            hit_positions.append([x+1, y])
            board[x+1][y] = 'X'
            return
        board[x+1][y] = '0'
        return

    # if there is no hit, pop from stack
    hit_positions.pop()
    make_move(board)            # when there is no valid moves around the current hit_position, restart making move
        
    


def check_valid_coordinates(board,x,y):
    if(x >= 0 and x < len(board) and y >= 0 and y < len(board)):
        return True
    return False


def check_if_hit(board, x, y):
    if board[x][y] in ['1', '2', '3', '4', '5']:
        return True
    return False






def play(board):
    moves = 0

    while True:
        make_move(board)
        moves += 1
        print('moves: ',moves)
        # utils.display_board(board)

        if utils.check_if_won(board):
            break

    # print('hunt play done')
    utils.display_board(board)
    return moves


if __name__ == '__main__':
    moves = []
    for i in range(30):
        board = utils.get_board('random_ships')
        moves.append( play(board) )
    print(moves)
