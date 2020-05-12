# battleship ai

import copy
import pprint
import random

import utils
import random_algo
import hunt_algo
import probability_density
import deep_reinforcement







def play_solo(algo='random', no_of_plays=1):
    board = utils.get_board('random_ships')





def play():
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




def test_play():
    print('starting test play...')
    board = utils.get_board(type='random_ships', size=10)
    m = hunt_algo.play(board)
    print("moves: ", m)









def main():
    test_play()
    # utils.display_board(utils.get_board(type='random_ships'))


if __name__ == '__main__':
    main()










# random

# hunt algorithm
    # explore nearby after hit

# probability density function / heatmap
    # https://cliambrown.com/battleship/methodology.php#options

# deep reinforcement learning
    # https://www.efavdb.com/battleship



# https://paulvanderlaken.com/2019/01/21/beating-battleships-with-algorithms-and-ai/
