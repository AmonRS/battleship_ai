# battleship ai

import copy
import pprint
import random
import statistics

import utils
import random_algo
import hunt_algo
import probability_density
import deep_reinforcement
import rl





# def play_solo(no_of_plays=1):
    
#     random_moves = []
#     hunt_moves = []
#     prob_moves = []
#     deep_moves = []

#     for i in range(no_of_plays):

#         board = utils.get_board('random_ships')

#         random_moves.append( random_algo.play( copy.deepcopy(board) ) )
#         hunt_moves.append( hunt_algo.play( copy.deepcopy(board) ) )
#         # prob
#     # deep r
#     # deep_moves.append( deep_reinforcement.play(board), no_of_plays )
#     rl_moves = rl.play(no_of_plays)

#     print ('average moves for each algo to win: ')
#     print('random: ', statistics.mean(random_moves), '---', random_moves)
#     print('hunt: ', statistics.mean(hunt_moves), '---', hunt_moves)
#     # print('prob: ', statistics.mean(prob_moves))
#     # print('deep: ', statistics.mean(deep_moves))
#     print('RL: ', statistics.mean(rl_moves), '---', rl_moves)




def play_tournament():
    pass

def play_ai_vs_player():
    # lets not do this one :(
    pass






def test_hunt():
    board = utils.get_board('random_ships')
    m = hunt_algo.play( board )
    print('hunt moves: ', m)

def test_rl():
    print('starting deep_r test play...')

    rl.play(10)

    # print("moves: ", m)
    print('rl testing done ...')


def test_prob():
    board = utils.get_board('random_ships')
    m = probability_density.play(board)
    print('total moves: ', m)





def main():
    # play_solo(30)
    # test_rl()
    test_prob()


if __name__ == '__main__':
    main()
