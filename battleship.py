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

    print ( 'average moves for each algo to win: ' )
    print( 'random: ', statistics.mean(random_moves), '---', random_moves )
    print( 'hunt: ', statistics.mean(hunt_moves), '---', hunt_moves )
    print( 'prob: ', statistics.mean(prob_moves), '---', prob_moves )
    print( 'RL: ', statistics.mean(rl_moves), '---', rl_moves )

    plot_results( random_moves, hunt_moves, prob_moves, rl_moves )



def play_tournament(no_of_plays=1):
    # 6 DIFFERENT GAME COMPETITIONS

    # random vs. hunt
    random_vs_hunt = { 'random': 0, 'hunt': 0 }
    for i in range(no_of_plays):
        rh_random_moves = random_algo.play( utils.get_board('random_ships') )
        rh_hunt_moves = hunt_algo.play( utils.get_board('random_ships') )
        if rh_random_moves < rh_hunt_moves:
            random_vs_hunt['random'] += 1
        elif rh_hunt_moves < rh_random_moves:
            random_vs_hunt['hunt'] += 1

    # random vs. prob
    random_vs_prob = { 'random': 0, 'prob': 0 }
    for i in range(no_of_plays):
        rp_random_moves = random_algo.play( utils.get_board('random_ships') )
        rp_prob_moves = probability_density.play( utils.get_board('random_ships') )
        if rp_random_moves < rp_prob_moves:
            random_vs_prob['random'] += 1
        elif rp_prob_moves < rp_random_moves:
            random_vs_prob['prob'] += 1
    
    # hunt vs. prob
    hunt_vs_prob = { 'hunt': 0, 'prob': 0 }
    for i in range(no_of_plays):
        hp_hunt_moves = hunt_algo.play( utils.get_board('random_ships') )
        hp_prob_moves = probability_density.play( utils.get_board('random_ships') )
        if hp_hunt_moves < hp_prob_moves:
            hunt_vs_prob['hunt'] += 1
        elif hp_prob_moves < hp_hunt_moves:
            hunt_vs_prob['prob'] += 1

    # random vs. RL
    random_vs_rl = {'random':0, 'rl':0}
    rrl_random_moves = []
    for i in range(no_of_plays):
        rrl_random_moves.append( random_algo.play( utils.get_board('random_ships') ) )
    rrl_rl_moves = rl.play( no_of_plays )
    for i in range(no_of_plays):
        if rrl_random_moves[i] < rrl_rl_moves[i]:
            random_vs_rl['random'] += 1
        elif rrl_rl_moves[i] < rrl_random_moves[i]:
            random_vs_rl['rl'] += 1

    # hunt vs. RL
    hunt_vs_rl = {'hunt':0, 'rl':0}
    hrl_hunt_moves = []
    for i in range(no_of_plays):
        hrl_hunt_moves.append( hunt_algo.play( utils.get_board('random_ships') ) )
    hrl_rl_moves = rl.play( no_of_plays )
    for i in range(no_of_plays):
        if hrl_hunt_moves[i] < hrl_rl_moves[i]:
            hunt_vs_rl['hunt'] += 1
        elif hrl_rl_moves[i] < hrl_hunt_moves[i]:
            hunt_vs_rl['rl'] += 1

    # prob vs. RL
    prob_vs_rl = {'prob':0, 'rl':0}
    prl_prob_moves = []
    for i in range(no_of_plays):
        prl_prob_moves.append( probability_density.play( utils.get_board('random_ships') ) )
    prl_rl_moves = rl.play( no_of_plays )
    for i in range(no_of_plays):
        if prl_prob_moves[i] < prl_rl_moves[i]:
            prob_vs_rl['prob'] += 1
        elif prl_rl_moves[i] < prl_prob_moves[i]:
            prob_vs_rl['rl'] += 1



    print('TOURNAMENT GAME BETWEEN ALGORITHMS')
    print('\t Random vs. Hunt')
    print('\t\t random: ', random_vs_hunt['random'], 'wins')
    print('\t\t hunt: ', random_vs_hunt['hunt'], 'wins')

    print('\t Random vs. Probability Density')
    print('\t\t random: ', random_vs_prob['random'], 'wins')
    print('\t\t probability: ', random_vs_prob['prob'], 'wins')

    print('\t Hunt vs. Probability Density')
    print('\t\t hunt: ', hunt_vs_prob['hunt'], 'wins')
    print('\t\t probability: ', hunt_vs_prob['prob'], 'wins')

    print('\t Random vs. Reinforcement Learning')
    print('\t\t random: ', random_vs_rl['random'], 'wins')
    print('\t\t learning: ', random_vs_rl['rl'], 'wins')

    print('\t Hunt vs. Reinforcement Learning')
    print('\t\t hunt: ', hunt_vs_rl['hunt'], 'wins')
    print('\t\t learning: ', hunt_vs_rl['rl'], 'wins')

    print('\t Probability Density vs. Reinforcement Learning')
    print('\t\t probability: ', prob_vs_rl['prob'], 'wins')
    print('\t\t learning: ', prob_vs_rl['rl'], 'wins')



def play_ai_vs_player():
    # lets not do this one :(
    pass





def test_prob():
    board = utils.get_board('random_ships')
    m = probability_density.play(board)
    print('total moves: ', m)





def main():
    # play_solo(30)
    play_tournament(5)
    # test_rl()
    # test_prob()

if __name__ == '__main__':
    main()
