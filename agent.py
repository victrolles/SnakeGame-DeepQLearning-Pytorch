import math
import torch
import random
import numpy as np
from collections import deque
from game import GameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import Plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 #0.001
SIZE = 40

RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

class Agent:

    def __init__(self):
        self.epoch = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


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

        # print("state: ", state)
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, average_score):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.epoch # 80
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon: # 200
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

class Training:
    def __init__(self):
        self.plot_scores = []
        self.plot_mean_scores = []
        self.mean_10_scores = 0
        self.plot_mean_10_scores = []
        self.plot_train_loss = []
        self.total_score = 0
        self.record = 0
        self.agent = Agent()
        self.game = GameAI()
        self.plotC = Plot()
        # self.load_nn()

    def train(self):
        while True:
            # get old state
            # print("state_old")
            state_old = self.agent.get_state(self.game)

            # get move
            final_move = self.agent.get_action(state_old, self.mean_10_scores)

            # perform move and get new state
            reward, done, score = self.game.play(final_move)
            # print("state_new")
            state_new = self.agent.get_state(self.game)

            # train short memory
            self.agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            self.agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                self.game.reset()
                self.agent.epoch += 1
                self.agent.train_long_memory()

                if score > self.record:
                    self.record = score
                    self.save()

                print('Game', self.agent.epoch, 'Score', score, 'Record:', self.record)

                # information for plotting
                self.plot_scores.append(score)

                self.total_score += score

                mean_score = self.total_score / self.agent.epoch
                mean_10_score = np.mean(self.plot_scores[-10:])
                self.mean_10_scores = np.ceil(mean_score)

                self.plot_mean_scores.append(mean_score)
                self.plot_mean_10_scores.append(mean_10_score)

                self.plot_train_loss.append(self.agent.trainer.loss.item())

                self.plotC.update_plot(self.plot_scores, self.plot_mean_scores, self.plot_mean_10_scores, self.plot_train_loss)

    def save(self):
        torch.save({
            'epoch': self.agent.epoch,
            'model_state_dict': self.agent.model.state_dict(),
            'optimizer_state_dict': self.agent.trainer.optimizer.state_dict(),
            'loss': self.agent.trainer.loss,
            'plot_scores': self.plot_scores,
            'plot_mean_scores': self.plot_mean_scores,
            'plot_mean_10_scores': self.plot_mean_10_scores,
            'plot_train_loss': self.plot_train_loss,
            'record': self.record,
            'total_score': self.total_score,
            'timer': self.game.time + self.game.saved_time,
        }, 'model/model.pth')

    def load_nn(self):
        checkpoint = torch.load('model/model.pth')
        self.agent.epoch = checkpoint['epoch']
        self.agent.model.load_state_dict(checkpoint['model_state_dict'])
        self.agent.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.agent.trainer.loss = checkpoint['loss']
        self.plot_scores = checkpoint['plot_scores']
        self.plot_mean_scores = checkpoint['plot_mean_scores']
        self.plot_mean_10_scores = checkpoint['plot_mean_10_scores']
        self.plot_train_loss = checkpoint['plot_train_loss']
        self.record = checkpoint['record']
        self.total_score = checkpoint['total_score']
        self.game.saved_time = checkpoint['timer']
        self.agent.model.eval()


if __name__ == '__main__':
    training = Training()
    training.train()