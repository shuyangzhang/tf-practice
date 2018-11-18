# coding = utf-8

from RL.tic_tac_toe import TicTacToe
from RL.RL_brain import QLearningTable
import random
import time

# env = TicTacToe()

n_iterations = 300000
n_max_step = 9
discount_rate = 0.95
tie_reward = 0.999

actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]

total_result = {1: 0, -1: 0, tie_reward: 0}

if __name__ == "__main__":
    env = TicTacToe(tie_reward=tie_reward)
    RL = QLearningTable(actions=actions, q_table="ttt_q_table.csv", learning_rate=0.1)

    for iteration in range(n_iterations):
        env.reset()

        history = []
        while not env.done:

            if env.turn == 1:
                has_played = False
                while not has_played:
                    player = 1
                    observation = str(env.chessboard.flatten())
                    action = RL.choose_action(observation)
                    obs, reward, done, info = env.step(player, action)

                    has_played = env.info
                    if has_played:
                        observation_ = str(obs.flatten())
                        history.append([observation, action, reward, observation_])


            else:
                has_played = False
                while not has_played:
                    player = -1
                    action = random.randint(0, 8)
                    env.step(player, action)
                    has_played = env.info


        if env.reward == tie_reward:
            this_result = tie_reward
        else:
            this_result = -1 * env.turn

        # update reward here, because it can catch lose
        history[-1][2] = this_result

        # training
        for observation, action, reward, observation_ in history:
            RL.learn(observation, action, reward, observation_)

        total_result[this_result] += 1

        # print("------done-----")
        # print("winner is {}".format(this_result))
        # print(env.chessboard)
        # print("---------------")

        # time.sleep(3)

        if iteration % 500 == 0:
            print("============")
            print("iter {}".format(iteration))
            print("============")
            print("win rate is {}".format(total_result[1] / (total_result[1] + total_result[tie_reward] + total_result[-1])))
            print("tie rate is {}".format(total_result[tie_reward] / (total_result[1] + total_result[tie_reward] + total_result[-1])))
            print("lose rate is {}".format(total_result[-1] / (total_result[1] + total_result[tie_reward] + total_result[-1])))
            print("============")

            # time.sleep(3)

    del RL
