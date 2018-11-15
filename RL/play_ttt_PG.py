# coding = utf-8

import tensorflow as tf
import random
import numpy as np

from RL.tic_tac_toe import TicTacToe

n_inputs = 9
n_hidden1 = 50
n_hidden2 = 50
n_outputs = 9


X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden1 = tf.layers.dense(X,
                          units=n_hidden1,
                          activation=tf.nn.relu,
                          use_bias=True,
                          kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.02),
                          bias_initializer=tf.constant_initializer(0.01),
                          )
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, use_bias=True,
                          kernel_initializer=tf.truncated_normal_initializer(mean=0., stddev=0.02),
                          bias_initializer=tf.constant_initializer(0.01),
                          )
logits = tf.layers.dense(hidden2, n_outputs, activation=None)
prob = tf.nn.softmax(logits)
action = tf.multinomial(logits, num_samples=1)
log_prob = tf.nn.log_softmax(logits, axis=1)
# log_prob = logits - tf.log(tf.reduce_sum(tf.exp(logits), axis=1))

return_placeholders = tf.placeholder(tf.float32, [None, 1])
action_placeholders = tf.placeholder(tf.int32, [None, 1])
actions = tf.one_hot(action_placeholders, depth=9)
# tf.one_hot(action, depth=9, axis=-1)
policy_gradient_with_return = - tf.reduce_mean(tf.multiply(
    return_placeholders,
    tf.reduce_sum(actions*log_prob, axis=1, keepdims=True)
))

LR = 0.001

training_op = tf.train.AdamOptimizer(LR).minimize(policy_gradient_with_return)

# optimizer = tf.train.AdamOptimizer(LR)
#
# grads_and_vars = optimizer.compute_gradients(policy_gradient)
#
# gradients = [grad for grad, variable in grads_and_vars]
#
# gradient_placeholders = []
# grads_and_vars_feed = []
#
# for grad, variable in grads_and_vars:
#     gradient_placeholder = tf.placeholder(tf.float32, shape=grad.shape())
#     gradient_placeholders.append(gradient_placeholder)
#     grads_and_vars_feed.append((gradient_placeholder, variable))
#
# training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + discount_rate * cumulative_rewards
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

n_iterations = 50000
n_max_step = 9
n_games_per_update = 10
discount_rate = 0.95

env = TicTacToe()

total_result = {
    -1: 0,
    0.5: 0,
    1: 0
}

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(n_iterations):
        # all_rewards = []
        # all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            # current_gradients = []
            current_act = []
            current_X = []
            current_X.append(env.chessboard.reshape(1, n_inputs))
            env.reset()

            while not env.done:
                # print(env.chessboard)
                if env.turn == 1:
                    has_played = False
                    current_X.append(env.chessboard.reshape(1, n_inputs).copy())
                    while not has_played:
                        player = 1
                        # action_val, gradients_val = sess.run(
                        #     [action, gradients],
                        #     feed_dict={X: env.chessboard.reshape(1, n_inputs)}
                        # )
                        action_val = sess.run(
                            action,
                            feed_dict={X: env.chessboard.reshape(1, n_inputs)}
                        )
                        obs, reward, done, info = env.step(player, action_val)

                        has_played = env.info
                        if has_played:
                            current_rewards.append(reward)
                            current_act.append(action_val)

                        # current_gradients.append(gradients_val)
                        # has_played = env.info
                else:  ## env.turn == -1
                    has_played = False
                    while not has_played:
                        player = -1
                        input_num = random.randint(0, 8)
                        env.step(player, input_num)
                        has_played = env.info

            ## end while, now this game is done
            if env.reward == 0.5:
                reward = 0.5
            else:
                reward = -1 * env.turn
            total_result[reward] +=1
            current_rewards[-1] = reward
            # print(current_X)
            current_X = current_X[:-1]
            # all_rewards.append(current_rewards)
            # all_gradients.append(current_gradients)

            # current_actions = [tf.one_hot(act, depth=9, axis=-1) for act in current_act]
            current_act = [int(act) for act in current_act]
            # print(current_act)
            # current_actions = tf.one_hot(current_act, depth=9)
            current_returns = discount_rewards(current_rewards, discount_rate)

            # print(current_returns)
            # print(current_act)
            # print(current_X)

            sess.run(training_op, feed_dict={
                X: np.array(current_X).flatten().reshape(-1, 9),
                action_placeholders: np.array(current_act).flatten().reshape(-1, 1),
                return_placeholders: np.array(current_returns).flatten().reshape(-1, 1)
            })

        # end for n_game_updates
        # all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        # feed_dict = {}
        # for var_index, grad_placeholder in enumerate(gradient_placeholder):
        #     mean_gradients = np.mean(
        #         [
        #             reward * all_gradients[game_index][step][var_index]
        #             for game_index, rewards in enumerate(all_rewards)
        #             for step, reward in enumerate(rewards)
        #         ],
        #         axis=0
        #     )
        #     feed_dict[grad_placeholder] = mean_gradients
        # sess.run(training_op, feed_dict=feed_dict)

        if iteration % 100 == 0:
            print("============")
            print("iter {}".format(iteration))
            print("============")
            print("win rate is {}".format(total_result[1] / (total_result[1] + total_result[0.5] + total_result[-1])))
            print("tie rate is {}".format(total_result[0.5] / (total_result[1] + total_result[0.5] + total_result[-1])))
            print("lose rate is {}".format(total_result[-1] / (total_result[1] + total_result[0.5] + total_result[-1])))
            print("============")