# https://towardsdatascience.com/an-artificial-intelligence-learns-to-play-battleship-ebd2cf9adb01
# https://bitbucket.org/mpi4py/mpi4py/issues/67/from-mpi4py-import-mpi-dont-work

# $ pip install gym stable-baselines[mpi]
# install Microsoft MPI

import gym
from gym import spaces
from stable_baselines.common.env_checker import check_env
import numpy as np




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
        """
        Important: the observation must be a numpy array
        :return: (np.array) 
        """
        
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
        # board (i,j)
        ## ------------>j
        ## | (0,0) | (0,1) | (0,2) | |
        ## | (1,0) | (1,1) | (1,2) | |
        ##                           v i
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







def play(no_of_plays):
    pass