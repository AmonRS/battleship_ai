import random
import utils
import copy

# This algorithm solely looks at the probability density function
# Even after it gets a hit, it still follows the function insteaed of looking for the surrounding squares

possible_locations = []
remaining_ships = [2,3,3,4,5]               # stores the length of the ships
# incompatible_locations = []
predicted_ships_locations = []              # didn't need to make it global, but it is faster (but probably not memory efficient) this way
hit_locations = []
probability_board = []


def make_move(board):
    global probability_board
    print("making move")
    while(True):
        next_move = max_2d_list(probability_board)
        if (probability_board[next_move[0]][next_move[1]]==0):
            make_random_move(board)
            break
        if (board[next_move[0]][next_move[1]] in ['~']):
            print("move with highest probability: ",next_move[0], next_move[1], "---------", board[next_move[0]][next_move[1]])
            board[next_move[0]][next_move[1]] = '0'
            break
        elif (board[next_move[0]][next_move[1]] in ['1','2','3','4','5']):
            print("move with highest probability: ",next_move[0], next_move[1], "---------", board[next_move[0]][next_move[1]])
            board[next_move[0]][next_move[1]] = 'X'
            break
        else:
            # if the probability board provides a non zero value for a position with either 'X' or '0' (already attempted)
            # set the probability to be 0 and choose the point again
            probability_board[next_move[0]][next_move[1]] = 0
        

def make_random_move(board):

    while True:
        x = random.randint(0, len(board)-1)
        y = random.randint(0, len(board)-1)

        if board[x][y] in ['1','2','3','4','5']:                    # hit a ship
            print('random move: ',x,y, '-----',board[x][y])
            board[x][y] = 'X'
            break
        elif board[x][y] in ['~']:                                  # hit water
            print('random move: ',x,y, '-----',board[x][y])
            board[x][y] = '0'
            break  


def max_2d_list(prob_board):
    a = -1                  # can set this to be any negative number as the values in probability board is never negative
    max_index = [0,0]
    for x in range(len(prob_board)):
        for y in range(len(prob_board)):
            if(prob_board[x][y]>a):
                a = prob_board[x][y]
                max_index = [x,y]
    return max_index


def prepare_to_make_move(board):
    global probability_board
    global possible_locations
    global predicted_ships_locations

    print("\n\n\npreparing to make move")

    # clearing the probability board before making every move
    probability_board = []
    for i in range(len(board)):
        probability_board.append([0]*len(board))

    # filling up the possibility list
    possible_locations = []
    for ship in range (len(remaining_ships)):
        possible_locations.append([])
        for x_point in range(len(board)):
            for y_point in range(len(board)):
                for orientation in range(2):                # horizontal = 0, vertical = 1
                    if(check_overhang_or_miss_on_position(board, x_point, y_point, ship, orientation)):
                        point = [x_point,y_point]
                        possible_locations[ship].append(point)
    

    # --------------Incomplete and probably unnecessary check-----------------------
    # # filling up the incompatibiliy list
    # for ship in range(len(remaining_ships)):
    #     for loc in possible_locations[ship]:
    # ------------------------------------------------------------------------------


    # generate a 1000, for a board of 10*10, (the more the better) positions of the ships, using the possibility list
    for i in range (pow(len(board),3)):
        # # IF THE ALGORITHM IS TOO SLOW, LET THE INVALID POSITIONS COUNT AS THE UNIVERSAL SPACE
        # # RIGHT NOW, IT LOOKS FOR 1000 VALID POSITIONS
        # # WHEN MADE FASTER, IT WILL LOOK 1000 POSITIONS (NO MATTER VALID OR NOT)
        # Now the algorithm looks for 1000 positions instead of 1000 valid positions
        # while(True):                                           # loop 1
        predicted_ships_locations = []                      # empty the prediction list at the beginning of making a set of random ship positions
        valid_positions = True
        for ship in range(len(remaining_ships)):
            # while(True):                                    # loop 2
                
            predicted_ships_locations.append([])

            index = random.randint(0,len(possible_locations[ship])-1)
            random_point = possible_locations[ship][index]
            orientation = random.randint(0,1)           # horizontal = 0, vertical = 1
            if(not generate_and_check_positions(board, random_point[0], random_point[1], orientation, ship)):
                # break                                   # break out of loop 2
                valid_positions = False

        if(not valid_positions):
            continue

        # print("got a set -- ")
        if(not check_hit_locations_predicted):
            # break                                           # break out of loop 1
            continue
        
        # now the positions stored in predicted_ships_locations are used to fill the probability board
        for ship_positions in predicted_ships_locations:
            for point in ship_positions:
                probability_board[point[0]][point[1]] += 1

    print("preperation ended")
    make_move(board)
        

# check if the hit_locations are all in the predicted ships
# True return means all the hit_locations are in the orientation
def check_hit_locations_predicted():
    print("checking for all the hits")
    for hit_loc in hit_locations:
        if(not check_single_hit_location(hit_loc)):
            return False
    return True

# True return means the hit_loc is in at least one ship's location (the other function ensures its one)
def check_single_hit_location(hit_loc):
    for ship_positions in predicted_ships_locations:
        if hit_loc in ship_positions:                   # generate_and_check_positions ensures that there is only one ship with one location
            return True
    return False
                        

# generates the position of a whole ship starting from the passed point
# also checks if this starting position intersects other ships
# True return means the starting coordinates are valid
def generate_and_check_positions(board, x, y, orientation, ship):
    global predicted_ships_locations

    # the other coordinates of the ships should be larger than (x, y) if possible
    pivot = 0                           # pivot is the index (in the loops below) which has an invalid location
    dir = 1                             # dir is 1 when increasing and -1 when decreasing the coordinate
    if (orientation == 0):                          # change y
        for i in range(remaining_ships[ship]):
            current_y = y+pivot+dir*i
            if(present_in_predicted([x,current_y])):
                return False
            predicted_ships_locations[ship].append([x,current_y])
            if(current_y+dir < 0 or current_y+dir >= len(board) or board[x][current_y+dir]=='0'):
                if(dir == -1):
                    return False            # if the direction is already opposite, then there are '0' in both sides, so return false
                pivot = i
                dir = -1
    else:
        for i in range(remaining_ships[ship]):
            current_x = x+pivot+dir*i
            if(present_in_predicted([current_x,y])):
                return False
            predicted_ships_locations[ship].append([current_x,y])
            if(current_x+dir < 0 or current_x+dir>=len(board) or board[current_x+dir][y]=='0'):
                if(dir == -1):
                    return False            # same logic as above
                pivot = i
                dir = -1
    return True

# True means start the loop again
def present_in_predicted(given_point):
    for ship_positions in predicted_ships_locations:
        if given_point in ship_positions:
            return True
    return False



# returns false if there is an overhang or a miss in the position
# true means put the coordinate on the 'possible' list
def check_overhang_or_miss_on_position(board,x,y, ship, orientation):       # I DON'T KNOW WHY I HAD TO PASS A GLOBAL VARIBALE
    # if the location is already in the array, dont add it
    if [x,y] in possible_locations[ship]:
        return False
    if(orientation == 0):       # horizontal
        if(not check_surrounding_horizontal(board, x, y, ship)):
            return False
    if(orientation == 1):
        if(not check_surrounding_vertical(board,x,y,ship)):
            return False
    return True



def check_surrounding_horizontal(board, x, y, ship):
    count = 0
    for index in range(y):
        if(board[x][y-(index+1)]!='0'):
            count+=1
        else:
            break
    
    for index in range(len(board)-y):
        if(board[x][y+index]!='0'):
            count+=1
        else:
            break
    
    if(count>=remaining_ships[ship]):
        return True
    else:
        return False
        

def check_surrounding_vertical(board, x, y, ship):
    count = 0
    for index in range(x):
        if(board[x-(index+1)][y]!='0'):
            count+=1
        else:
            break
    
    for index in range(len(board)-x):
        if(board[x+index][y]!='0'):
            count+=1
        else:
            break
    
    if(count>=remaining_ships[ship]):
        return True
    else:
        return False





# same function in all the algorithms
def play(board):
    global remaining_ships
    moves = 0

    while True:
        prepare_to_make_move(board)
        moves += 1
        print('moves: ',moves)
        # utils.display_board(board)

        # resetting remaining ships
        remaining_ships = []
        floating_ships = utils.get_floating_ships(board)
        for ship_number in floating_ships:
            remaining_ships.append(utils.ships[ship_number-1][2])

        if utils.check_if_won(board):
            break

    # print('hunt play done')
    utils.display_board(board)
    return moves





if __name__ == '__main__':
    m = play( utils.get_board('random_ships') )
    print('moves: ',m)