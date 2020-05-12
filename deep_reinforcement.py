# import tensorflow as tf
# import numpy as np
# # %matplotlib inline            # for jupyter notebooks so the plot shows in the notebook
# import pylab





# BOARD_SIZE = 10
# # SHIP_SIZE = 3

# hidden_units = BOARD_SIZE
# output_units = BOARD_SIZE

# input_positions = tf.placeholder( tf.float32, shape=(1, BOARD_SIZE) )
# labels = tf.placeholder( tf.int64 )
# learning_rate = tf.placeholder( tf.float32, shape=[] )

# # Generate hidden layer
# W1 = tf.Variable( tf.truncated_normal([BOARD_SIZE, hidden_units],stddev=0.1 / np.sqrt(float(BOARD_SIZE))) )
# b1 = tf.Variable( tf.zeros([1, hidden_units]) )
# h1 = tf.tanh( tf.matmul(input_positions, W1) + b1 )

# # Second layer -- linear classifier for action logits
# W2 = tf.Variable( tf.truncated_normal([hidden_units, output_units],stddev=0.1 / np.sqrt(float(hidden_units))) )
# b2 = tf.Variable( tf.zeros([1, output_units]) )
# logits = tf.matmul(h1, W2) + b2
# probabilities = tf.nn.softmax( logits )

# init = tf.initialize_all_variables()
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='xentropy')
# train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# # Start TF session
# sess = tf.Session()
# sess.run(init)






# TRAINING = True
# def play_game(training=TRAINING):
#     """ Play game of battleship using network."""

#     # # Select random location for ship
#     # ship_left = np.random.randint(BOARD_SIZE - SHIP_SIZE + 1)
#     # ship_positions = set(range(ship_left, ship_left + SHIP_SIZE))

#     # Initialize logs for game
#     board_position_log = []
#     action_log = []
#     hit_log = []

#     # Play through game
#     current_board = [[-1 for i in range(BOARD_SIZE)]]

#     while sum(hit_log) < SHIP_SIZE:
#         board_position_log.append([[i for i in current_board[0]]])
#         probs = sess.run([probabilities], feed_dict={input_positions:current_board})[0][0]
#         probs = [p * (index not in action_log) for index, p in enumerate(probs)]
#         probs = [p / sum(probs) for p in probs]
#         if training == True:
#             bomb_index = np.random.choice(BOARD_SIZE, p=probs)
#         else:
#             bomb_index = np.argmax(probs)
#         # update board, logs
#         hit_log.append(1 * (bomb_index in ship_positions))
#         current_board[0][bomb_index] = 1 * (bomb_index in ship_positions)
#         action_log.append(bomb_index)
#     return board_position_log, action_log, hit_log








# def rewards_calculator(hit_log, gamma=0.5):
#     """ Discounted sum of future hits over trajectory"""
#     hit_log_weighted = [(item -
#     float(SHIP_SIZE - sum(hit_log[:index])) / float(BOARD_SIZE - index)) * (
#         gamma ** index) for index, item in enumerate(hit_log)]
#     return [((gamma) ** (-i)) * sum(hit_log_weighted[i:]) for i in range(len(hit_log))]









# game_lengths = []
# TRAINING = True # Boolean specifies training mode
# ALPHA = 0.06 # step size

# for game in range(10000):
#     board_position_log, action_log, hit_log = play_game(training=TRAINING)
#     game_lengths.append(len(action_log))
#     rewards_log = rewards_calculator(hit_log)
#     for reward, current_board, action in zip(rewards_log, board_position_log, action_log):
#         # Take step along gradient
#         if TRAINING:
#             sess.run([train_step],
#                 feed_dict={input_positions:current_board, labels:[action], learning_rate:ALPHA * reward})














# def play( board, no_of_plays ):
#     # print('starting deep reinforcement play')

#     moves = 0


#     # print('deep reinforment play done')
#     return moves