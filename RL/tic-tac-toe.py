# coding = utf-8

import numpy as np

class TicTacToe(object):
    def __init__(self):
        self.reset()

    def step(self, player, position):
        position = self._position_to_xy(position)
        self.info = self._is_valid(player, position)

        if self.info:
            self.turn *= -1
            self.chessboard[position] = player
            self._is_done()
            return self.chessboard, self.reward, self.done, self.info

        else:
            return self.chessboard, self.reward, self.done, self.info

    def reset(self):
        self.chessboard = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])

        ## offensive position is 1, defensive position is -1
        self.turn = 1
        self.reward = 0
        self.done = False
        self.info = True  ## is True is this step is valid

    def _position_to_xy(self, position):
        position_num = int(position)
        return (position_num // 3, position_num % 3)


    def _is_valid(self, player, position):
        return (not self.chessboard[position]) and (player == self.turn)

    def _is_done(self):
        if 0 not in self.chessboard:
            self.reward = 0.5
            self.done = True
        else:
            if 3 in np.sum(self.chessboard, axis=0) or 3 in np.sum(self.chessboard, axis=1):
                self.reward = 1
                self.done = True
            elif -3 in np.sum(self.chessboard, axis=0) or -3 in np.sum(self.chessboard, axis=1):
                self.reward = 1
                self.done = True
            elif self.chessboard[0, 0] == self.chessboard[1, 1] == self.chessboard[2, 2] != 0:
                self.reward = 1
                self.done = True
            elif self.chessboard[0, 2] == self.chessboard[1, 1] == self.chessboard[2, 0] != 0:
                self.reward = 1
                self.done = True
            else:
                pass


if __name__ == "__main__":
    import random

    env = TicTacToe()
    while not env.done:
        print(env.chessboard)
        print("reward is {}".format(env.reward))
        if env.turn == 1:
            has_played = False
            while not has_played:
                player = 1
                input_num = int(input("请下棋："))
                # input_num = random.randint(0,8)
                # position = (input_num // 3, input_num % 3)
                env.step(player, input_num)
                has_played = env.info
        else:
            has_played = False
            while not has_played:
                player = -1
                input_num = random.randint(0,8)
                # position = (input_num // 3, input_num % 3)
                env.step(player, input_num)
                has_played = env.info

    print("------done-------")
    print("winner is {}".format(-1 * env.turn))
    print(env.chessboard)
    print("reward is {}".format(env.reward))

