# battleship ai

import copy
import matplotlib.pyplot as plt
import pprint
import random
import statistics

import utils
import random_algo
import hunt_algo
import probability_density
import rl










def plot_results(random_moves=[], hunt_moves=[], prob_moves=[], rl_moves=[]):


    plt.plot( [ i+1 for i in range(len(random_moves)) ], random_moves )
    plt.plot( [ i+1 for i in range(len(hunt_moves)) ], hunt_moves )
    plt.plot( [ i+1 for i in range(len(prob_moves)) ], prob_moves )
    plt.plot( [ i+1 for i in range(len(rl_moves)) ], rl_moves )

    plt.legend(['random', 'hunt', 'probability', 'RL'], loc='upper left')
    plt.xlabel('games played')
    plt.ylabel('no. of moves to win')
    
    plt.show()









def play_solo(no_of_plays=1):
    
    random_moves = []
    hunt_moves = []
    prob_moves = []
    deep_moves = []

    for i in range(no_of_plays):

        board = utils.get_board('random_ships')

        random_moves.append( random_algo.play( copy.deepcopy(board) ) )
        hunt_moves.append( hunt_algo.play( copy.deepcopy(board) ) )
        prob_moves.append( probability_density.play( copy.deepcopy(board) ) )
    rl_moves = rl.play(no_of_plays)

    print ('average moves for each algo to win: ')
    print('random: ', statistics.mean(random_moves), '---', random_moves)
    print('hunt: ', statistics.mean(hunt_moves), '---', hunt_moves)
    print('prob: ', statistics.mean(prob_moves), '---', prob_moves)
    print('RL: ', statistics.mean(rl_moves), '---', rl_moves)

    plot_results(random_moves, hunt_moves, prob_moves, rl_moves)

def play_tournament():
    # lets not do this either :(
    pass

def play_ai_vs_player():
    # lets not do this one :(
    pass








def test_prob():
    board = utils.get_board('random_ships')
    m = probability_density.play(board)
    print('total moves: ', m)





def main():
    play_solo(30)
    # test_rl()
    # test_prob()


if __name__ == '__main__':
    main()
