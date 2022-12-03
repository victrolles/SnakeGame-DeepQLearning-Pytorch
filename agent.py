import torch
import numpy as np
# from collections import deque
from game import GameAI, Direction, Point
# from model import Linear_QNet, QTrainer
from helper import Plot

# MAX_MEMORY = 100_000
# BATCH_SIZE = 1000
# LR = 0.001 #0.001
SIZE = 40

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

class Agent:

    def __init__(self):
        # Learning parameters
        self.epoch = 0
        self.lr = 0.01
        self.epsilon = 0.1 # randomness
        self.discount = 0.9 # discount rate

        # Memory
        self.q_values = np.zeros((2**11,3), dtype=np.float32)

        # Others parameters
        # self.plot_scores = []
        # self.plot_mean_scores = []
        # self.mean_10_scores = 0
        # self.plot_mean_10_scores = []
        # self.total_score = 0
        # self.record = 0
        self.game = GameAI()
        # self.plotC = Plot()
        # self.load_nn()

    def get_state(self, game):
        head = game.head

        point_l = Point(head.x - SIZE, head.y)
        point_r = Point(head.x + SIZE, head.y)
        point_u = Point(head.x, head.y - SIZE)
        point_d = Point(head.x, head.y + SIZE)
        
        dir_l = game.snake.direction == Direction.LEFT
        dir_r = game.snake.direction == Direction.RIGHT
        dir_u = game.snake.direction == Direction.UP
        dir_d = game.snake.direction == Direction.DOWN

        danger_s = (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l)) or (dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d))
        danger_r = (dir_u and game.is_collision(point_r)) or (dir_d and game.is_collision(point_l)) or (dir_l and game.is_collision(point_u)) or (dir_r and game.is_collision(point_d))
        danger_l = (dir_d and game.is_collision(point_r)) or (dir_u and game.is_collision(point_l)) or (dir_r and game.is_collision(point_u)) or (dir_l and game.is_collision(point_d))

        food_l = game.food.x < game.head.x  # food left
        food_r = game.food.x > game.head.x  # food right
        food_u = game.food.y < game.head.y  # food up
        food_d = game.food.y > game.head.y  # food down

        state = [
            #direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location 
            food_l,
            food_r,
            food_u,
            food_d,

            # Possible direction
            danger_l,
            danger_r,
            danger_s
            ]

        print("state: ", state)
        return np.array(state, dtype=int)

    def get_action(self, state):
        # Epsilon greedy
        final_move = [0,0,0]

        if np.random.random() < self.epsilon:
            print("random")
            move = np.random.randint(3)
            final_move[move] = 1
        else:
            print("q_values")
            final_move = np.argmax(self.q_values[state])

        print("final_move: ", final_move)
        return np.array(final_move, dtype=int)

    def train(self):
        while True:
            # get old state
            # print("state_old")
            state_old = self.get_state(self.game)

            # get move
            final_move = self.get_action(state_old)

            # perform move and get new state
            reward, done, score = self.game.play(final_move)
            # print("state_new")
            state_new = self.get_state(self.game)

            # train short memory
            self.q_values[state_old, final_move] = self.q_values[state_old, final_move] + self.lr * (reward + self.discount * np.max(self.q_values[state_new]) - self.q_values[state_old, final_move])

            if done:
                # train long memory, plot result
                self.game.reset()
                self.epoch += 1

                if score > self.record:
                    self.record = score
                    # self.save()

                print('Game', self.epoch, 'Score', score, 'Record:', self.record)

                # information for plotting
                # self.plot_scores.append(score)

                # self.total_score += score

                # mean_score = self.total_score / self.epoch
                # mean_10_score = np.mean(self.plot_scores[-10:])
                # self.mean_10_scores = np.ceil(mean_score)

                # self.plot_mean_scores.append(mean_score)
                # self.plot_mean_10_scores.append(mean_10_score)

                # self.plotC.update_plot(self.plot_scores, self.plot_mean_scores, self.plot_mean_10_scores)

    # def save(self):
    #     torch.save({
    #         'epoch': self.epoch,
    #         'plot_scores': self.plot_scores,
    #         'plot_mean_scores': self.plot_mean_scores,
    #         'plot_mean_10_scores': self.plot_mean_10_scores,
    #         'record': self.record,
    #         'total_score': self.total_score,
    #         'timer': self.game.time + self.game.saved_time,
    #     }, 'model/model.pth')

    # def load_nn(self):
    #     checkpoint = torch.load('model/model.pth')
    #     self.epoch = checkpoint['epoch']
    #     self.plot_scores = checkpoint['plot_scores']
    #     self.plot_mean_scores = checkpoint['plot_mean_scores']
    #     self.plot_mean_10_scores = checkpoint['plot_mean_10_scores']
    #     self.record = checkpoint['record']
    #     self.total_score = checkpoint['total_score']
    #     self.game.saved_time = checkpoint['timer']

if __name__ == '__main__':
    agent = Agent()
    agent.train()