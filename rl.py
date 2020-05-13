# https://towardsdatascience.com/an-artificial-intelligence-learns-to-play-battleship-ebd2cf9adb01
# https://bitbucket.org/mpi4py/mpi4py/issues/67/from-mpi4py-import-mpi-dont-work

# $ pip install gym stable-baselines[mpi]
# install Microsoft MPI

import gym
from gym import spaces
from stable_baselines.common.env_checker import check_env
import numpy as np

from stable_baselines import DQN, PPO2, A2C, ACKTR, TRPO
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
import os
from stable_baselines.results_plotter import load_results, ts2xy
from tensorflow.keras.backend import clear_session #not sure if we need this but it does not hurt
import matplotlib.pyplot as plt

from stable_baselines.common.callbacks import BaseCallback




moves = []




# randomly places a ship on a board
def set_ship(ship, ships, board, ship_locs):

    grid_size = board.shape[0]
    
    done = False
    while not done:
        init_pos_i = np.random.randint(0, grid_size)
        init_pos_j = np.random.randint(0, grid_size)
                    
        # for a cruiser, if init_oos_i = 0, move forward horizontally (+1)
        # for a cruiser, if init_oos_j = 0, move downward vertically (+1)
        move_j = grid_size - init_pos_j - ships[ship]# horizontal
        if move_j > 0:
            move_j = 1
        else:
            move_j = -1
        move_i = grid_size - init_pos_i - ships[ship] # vertical
        if move_i > 0:
            move_i = 1
        else:
            move_i = -1
        # choose if placing ship horizontally or vertically
        choice_hv = np.random.choice(['h', 'v']) # horizontal, vertical
        if choice_hv == 'h': #horizontal
            j = [(init_pos_j + move_j*jj) for jj in range(ships[ship])]
            i = [init_pos_i for ii in range(ships[ship])]
            pos = set(zip(i,j))     
            if all([board[i,j]==0 for (i,j) in pos]):
                done = True
        elif choice_hv == 'v':
            i = [(init_pos_i + move_i*ii) for ii in range(ships[ship])]
            j = [init_pos_j for jj in range(ships[ship])]
            pos = set(zip(i,j))        
            #check if empty board in this direction
            if all([board[i,j]==0 for (i,j) in pos]):
                done = True
    # set ship - see convention
    for (i,j) in pos:
        board[i,j] = 1
        ship_locs[ship].append((i,j))
    
    return board, ship_locs

def board_rendering(grid_size, board):
    for i in range(grid_size):
        print("-"*(4*grid_size+2))
        for j in range(grid_size):
            current_state_value = board[i,j]
            current_state = ('S' if current_state_value==1 else ' ')
            print(" | ", end="")
            print(current_state, end='')
        print(' |')
    print("-"*(4*grid_size+2))




class BattleshipEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    
    def __init__(self, enemy_board, ship_locs, grid_size, ships):

        print('env __init__')
        
        super(BattleshipEnv, self).__init__()
        
        self.ships = ships
        self.grid_size = grid_size 
        self.cell = {'E': 0, 'X': 1, 'O': -1} 
        # boards, actions, rewards
        self.board = self.cell['E']*np.ones((self.grid_size, self.grid_size), dtype='int')

        # enemy_board must be encoded with 0: empy and 1: ship cell
        self.is_enemy_set = False
        self.enemy_board = enemy_board

        self.ship_locs = ship_locs

        if self.enemy_board is None:
            self.enemy_board = 0*np.ones((self.grid_size, self.grid_size), dtype='int')
            for ship in self.ships:
                self.ship_locs[ship] = []
                self.enemy_board, self.ship_locs = set_ship(ship, self.ships, self.enemy_board, self.ship_locs)
            self.is_enemy_set = True

        # reward discount
        self.rdisc = 0
        self.legal_actions = [] # legal (empty) cells available for moves
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.legal_actions.append((i,j))# this gets updated as an action is performed
        
        # Define action and observation space They must be gym.spaces objects In our case the action space is discrete: index of action
        self.action_space = spaces.Discrete(self.grid_size * self.grid_size)
        # The observation will be the state or configuration of the board
        self.observation_space = spaces.Box(low=-1, high=1,shape=(self.grid_size, self.grid_size), dtype=np.int)
        #Ex: print(spaces.Box(0,1, shape=(10,10)).high)

    def step(self, action):
        # action will be an index in action_space if from epsilon-greedy or from model prediction
                        
        # board situation before the action
        state = self.board.copy()        
        empty_cnts_pre, hit_cnts_pre, miss_cnts_pre = self.board_config(state)
        
        # action coordinates generated or predicted by the agent in the action_space
        i, j = np.unravel_index(action, (self.grid_size,self.grid_size))
        
        #print('action', action, 'coords', i, j)
        #print('legal_actions', self.legal_actions)
        
        # lose 1 point for any action
        reward = -1
        # assign a penalty for each illegal action used instead of a legal one
        if (i,j) not in self.legal_actions:
            reward -= 2*self.grid_size
            action_idx = np.random.randint(0,len(self.legal_actions))
            
            i,j = self.legal_actions[action_idx]                
            action = np.ravel_multi_index((i,j), (self.grid_size,self.grid_size))
        
        # set new state after performing action (scoring board is updated)
        self.set_state((i,j))
        # update legal actions and action_space
        self.set_legal_actions((i,j))

        # new state on scoring board - this includes last action
        next_state = self.board
               
        # board situation after action
        empty_cnts_post, hit_cnts_post, miss_cnts_post = self.board_config(next_state)

        # game completed?
        done = bool(hit_cnts_post == sum(self.ships.values()))
                    
        # reward for a hit
        if hit_cnts_post-hit_cnts_pre==1: 
            # Update hit counts and use it to reward
            r_discount = 1#0.5**self.rdisc
            rp = (self.grid_size*self.grid_size if done else self.grid_size)
            reward += rp*r_discount
            #print('HIT!!!')
                    
        reward = float(reward)
            
        #print('reward:', reward)
        # store the current value of the portfolio here
        info = {}

        return next_state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        
        self.board = self.cell['E']*np.ones((self.grid_size, self.grid_size), dtype='int')
        
        self.legal_actions = [] # legal (empty) cells available for moves
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.legal_actions.append((i,j))# this gets updated as an action is performed
               
        # generate a random board again if it was set randomly before
        if self.is_enemy_set:
            self.enemy_board = 0*np.ones((self.grid_size, self.grid_size), dtype='int')
            self.ship_locs = {}
            for ship in self.ships:
                self.ship_locs[ship] = []
                self.enemy_board, self.ship_locs = set_ship(ship, self.ships, self.enemy_board, self.ship_locs)

        self.rdisc = 0

        return self.board
 
    def render(self, mode='human'):
        # Render the environment to the screen
        for i in range(self.grid_size):
            print("-"*(4*self.grid_size+2))
            for j in range(self.grid_size):
                current_state_value = self.board[i,j]
                current_state = list(self.cell.keys())[list(self.cell.values()).index(current_state_value)]
                current_state = (current_state if current_state!='E' else ' ')
                print(" | ", end="")
                print(current_state, end='')
            print(' |')
        print("-"*(4*self.grid_size+2))
    

    def board_config(self, state):
        uni_states, uni_cnts = np.unique(state.ravel(), return_counts=True)
        empty_cnts = uni_cnts[uni_states==self.cell['E']]
        hit_cnts = uni_cnts[uni_states==self.cell['X']]
        miss_cnts = uni_cnts[uni_states==self.cell['O']]
        if len(empty_cnts)==0:
            empty_cnts = 0
        else:
            empty_cnts = empty_cnts[0]
        if len(hit_cnts)==0:
            hit_cnts = 0
        else:
            hit_cnts = hit_cnts[0]
        if len(miss_cnts)==0:
            miss_cnts = 0
        else:
            miss_cnts = miss_cnts[0]
        
        return empty_cnts, hit_cnts, miss_cnts

    # set board configuration and state value after player action
    def set_state(self, action):
        i , j = action
        if self.enemy_board[i,j]==1:
            self.board[i,j]=self.cell['X']
        else:
            self.board[i,j]=self.cell['O']

    # set legal actions (empty board locations)
    def set_legal_actions(self, action):
        if action in self.legal_actions:
            self.legal_actions.remove(action)





## to work with all policies. See next callback function

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward, moves
    # Print stats every step_interval calls
    if (n_steps + 1) % step_interval == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            # NOTE: when done is True, timesteps are counted and reported to the log_dir
            mean_reward = np.mean(y[-episode_interval:]) # mean reward over previous episode_interval episodes
            mean_moves = np.mean(np.diff(x[-episode_interval:])) # mean moves over previous episode_interval episodes
            print(x[-1], 'timesteps') # closest to step_interval step number
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f} - Last mean moves per episode: {:.2f}".format(best_mean_reward, 
                                                                                           mean_reward, mean_moves))
            moves.append( mean_moves )

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    # Returning False will stop training early
    return True
class SaveOnBestTrainingRewardCallback(BaseCallback):
   
    def __init__(self, check_freq: int, episode_interval: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_interval = episode_interval
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model.pkl')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate policy training performance
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # NOTE: when done is True, timesteps are counted and reported to the log_dir
                mean_reward = np.mean(y[-self.episode_interval:]) # mean reward over previous episode_interval episodes
                mean_moves = np.mean(np.diff(x[-self.episode_interval:])) # mean moves over previous 100 episodes
                if self.verbose > 0:
                    print(x[-1], 'timesteps') # closest to step_interval step number
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f} - Last mean moves per episode: {:.2f}".format(self.best_mean_reward, 
                                                                                                   mean_reward, mean_moves))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model")
                    self.model.save(self.save_path)

        return True






def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, window = 100, title='Learning Curve'):
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=window)
    y_moves = moving_average(np.diff(x), window = window) 
    # Truncate x
    x = x[len(x) - len(y):]
    x_moves = x[len(x) - len(y_moves):]

    title = 'Smoothed Learning Curve of Rewards (every ' + str(window) +' steps)'
    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()

    title = 'Smoothed Learning Curve of Moves (every ' + str(window) +' steps)'
    fig = plt.figure(title)
    plt.plot(x_moves, y_moves)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Moves')
    plt.title(title)
    plt.show()










# # validate environment with one ship (either random or user-defined) on 5x5 board

# # ships
# ships = {}
# ships['cruiser'] = 3

# grid_size = 5
# # for pre-determined board
# enemy_board = 0*np.ones((grid_size, grid_size), dtype='int')
# enemy_board[0,1] = 1
# enemy_board[1,1] = 1
# enemy_board[2,1] = 1
# ship_locs = {}
# ship_locs['cruiser'] = [(0,1),(1,1),(2,1)]
# print('going to create env ...')
# env = BattleshipEnv(enemy_board=enemy_board, ship_locs=ship_locs, grid_size=grid_size, ships=ships)
# # for random board
# #env = BattleshipEnv(enemy_board=None, ship_locs={}, grid_size=grid_size)
# # If the environment doesn't follow the interface, an error will be thrown
# check_env(env, warn=True)
# print('check env done :)')

# print(env.enemy_board)
# print(env.ship_locs)








def test_environment():
    # Test environment
    # ships
    ships = {}
    ships['destroyer'] = 2            # [ no. , name , size]
    ships['submarine'] = 3            # no. is used to identify ships on the board 
    ships['cruiser'] = 3 
    ships['battleship'] = 4 
    ships['carrier'] = 5

    grid_size=10

    env = BattleshipEnv(enemy_board=None, ship_locs={}, grid_size=grid_size, ships=ships)

    for ep in range(2):
        print('Episode', ep)
        obs = env.reset()
        #env.render()
        #print(env.enemy_board)
        done = False
        t = 0
        while not done:
            action = env.action_space.sample()
            i, j = np.unravel_index(action, (grid_size,grid_size))    
            print("Action {}".format(t + 1), i, j)
            obs, reward, done, _ = env.step(action)
            print('obs=', obs, 'reward=', reward, 'done=', done)
            env.render()
            t += 1
            if done:
                print("Goal reached!", "reward=", reward)







# clear_session()

# ships -- keep only one kind for 5x5 grid
ships = {}
ships['destroyer'] = 2            # [ no. , name , size]
ships['submarine'] = 3            # no. is used to identify ships on the board 
ships['cruiser'] = 3 
ships['battleship'] = 4 
ships['carrier'] = 5

grid_size=10

num_timesteps = 50000#1000000 # this is number of moves and not number of episodes

best_mean_reward, n_steps, step_interval, episode_interval = -np.inf, 0, 10000, 10000

log_dir = "./gym/"


def play_this_rl_thing():

    clear_session()

    # Instantiate the env
    env = BattleshipEnv(enemy_board=None, ship_locs={}, grid_size=grid_size, ships=ships)

    # wrap it
    
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, filename=log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    # Train the agent - Note: best model is not save in Callback function for PPO2; save manually
    model = A2C('MlpPolicy', env, verbose=0).learn(total_timesteps=num_timesteps, callback=callback)







def play(no_of_plays):
    global num_timesteps, moves
    num_timesteps = 10000 * no_of_plays

    play_this_rl_thing()

    print(moves)
    return moves





def test_rl(no_of_plays):
    play(no_of_plays)
if __name__ == '__main__':
    test_rl(7)