# battleship ai

import pprint
import random
import statistics

import utils
import random_algo
import hunt_algo
import probability_density
import deep_reinforcement





def play_solo(no_of_plays=1):
    
    random_moves = []
    hunt_moves = []
    prob_moves = []
    deep_moves = []

    for i in range(no_of_plays):

        baord = utils.get_board('random_ships')

        random_moves.append( random_algo.play(baord) )
        hunt_moves.append( hunt_algo.play(baord) )
        # prob
        # deep r

    print ('average moves for each algo to win: ')
    print('random: ', statistics.mean(random_moves))
    print('hunt: ', statistics.mean(hunt_moves))
    # print('prob: ', statistics.mean(prob_moves))
    # print('deep: ', statistics.mean(deep_moves))
    




def play(type_of_game='solo', num_of_games_each=1):

    if type_of_game == 'solo':
        play_solo(no_of_plays=num_of_games_each)



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
    m = random_algo.play(board)
    print("moves: ", m)









def main():
    test_play()

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
