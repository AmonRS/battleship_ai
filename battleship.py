# battleship ai

import copy
import pprint
import random

import random_algo



ships = [
    ['destroyer',2],
    ['submarine',3],
    ['cruiser',3],
    ['battleship',4],
    ['carrier',5]
]

def get_board(type='empty', size=10):
    board = []
    for i in range(size):
        board.append( [0]*size )

    if type == 'empty':
        return board
    elif type == 'random_ships':
        # place ships randomly
        return board









def play_solo(algo='random', no_of_plays=1):
    board = get_board('random_ships')













type_of_game = 'solo'

if type_of_game == 'solo':
    play_solo()
    # for several times:
        # randomly place the ships on the board
        # run an algorithm to solve it
        # get results (number to moves to win)
elif type_of_game == 'tournament':
    pass
    # tournament style
        # ...
elif type_of_game == 'player':
    pass
    # player vs. ai




















# random

# hunt algorithm
    # explore nearby after hit

# probability density function / heatmap
    # https://cliambrown.com/battleship/methodology.php#options

# deep reinforcement learning
    # https://www.efavdb.com/battleship












# https://paulvanderlaken.com/2019/01/21/beating-battleships-with-algorithms-and-ai/
