import torch
import random
import numpy as np
from collections import deque
from game import GameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
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
        self.model = Linear_QNet(16, 256, 3)
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
        
        nbr_free_r = 0
        nbr_free_l = 0
        nbr_free_u = 0
        nbr_free_d = 0

        if not dir_l:
            nbr_free_r = game.DFS(point_r, BLUE, occurence_test=True)

        if not dir_r:
            nbr_free_l = game.DFS(point_l, RED, occurence_test=True)

        if not dir_d:
            nbr_free_d = game.DFS(point_u, GREEN, occurence_test=True)

        if not dir_u:
            nbr_free_u = game.DFS(point_d, YELLOW, occurence_test=True)

        dir_cons_l = nbr_free_l == max(nbr_free_l, nbr_free_r, nbr_free_u, nbr_free_d)
        dir_cons_r = nbr_free_r == max(nbr_free_l, nbr_free_r, nbr_free_u, nbr_free_d)
        dir_cons_u = nbr_free_u == max(nbr_free_l, nbr_free_r, nbr_free_u, nbr_free_d)
        dir_cons_d = nbr_free_d == max(nbr_free_l, nbr_free_r, nbr_free_u, nbr_free_d)

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

            # Length of snake
            game.snake.length,

            # Number of free cells
            dir_cons_l,
            dir_cons_r,
            dir_cons_u,
            dir_cons_d
            ]

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

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 320 - self.epoch
        final_move = [0,0,0]
        if random.randint(0, 800) < self.epsilon:
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
        self.total_score = 0
        self.record = 0
        self.agent = Agent()
        self.game = GameAI()
        self.load_nn()

    def train(self):
        while True:
            # get old state
            state_old = self.agent.get_state(self.game)

            # get move
            final_move = self.agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = self.game.play(final_move)
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

                self.plot_scores.append(score)
                self.total_score += score
                mean_score = self.total_score / self.agent.epoch
                self.plot_mean_scores.append(mean_score)
                plot(self.plot_scores, self.plot_mean_scores)

    def save(self):
        torch.save({
            'epoch': self.agent.epoch,
            'model_state_dict': self.agent.model.state_dict(),
            'optimizer_state_dict': self.agent.trainer.optimizer.state_dict(),
            'loss': self.agent.trainer.loss,
            'plot_scores': self.plot_scores,
            'plot_mean_scores': self.plot_mean_scores,
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
        self.record = checkpoint['record']
        self.total_score = checkpoint['total_score']
        self.game.saved_time = checkpoint['timer']
        self.agent.model.eval()


if __name__ == '__main__':
    training = Training()
    training.train()