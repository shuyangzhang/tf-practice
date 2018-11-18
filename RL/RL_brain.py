# coding = utf-8

from RL.maze_env import Maze
import numpy as np
import pandas as pd


class QLearningTable(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.95, e_greedy=0.9, q_table=None):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.file_path = q_table
        try:
            self.q_table = pd.read_csv(self.file_path, index_col=0) if q_table else pd.DataFrame(columns=self.actions)
        except Exception as e:
            print(e)
            self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.ix[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != "terminal":
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )

    def __del__(self):
        if self.file_path:
            self.q_table.to_csv(self.file_path)
            print("q_table_file has been saved/updated.")
