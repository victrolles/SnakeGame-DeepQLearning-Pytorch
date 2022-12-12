import torch
import numpy as np
from collections import deque, namedtuple
from game import GameAI, Direction, Point, SIZE
import torch
import torch.nn as nn
import torch.optim as optim
from helper import Plot
import time

HISTORY_SIZE = 10_000
BATCH_SIZE = 1000

LR = 0.0005
GAMMA = 0.95

EPSILON_START = 1
EPSILON_END = 0.05
EPSILON_DECAY = 0.00001

SYNC_TARGET_EPOCH = 1000

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'done', 'next_state'))

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def _append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fully_connected_layers(x)
        

class DQN_trainer:
    def __init__(self, model_network, model_target_network, exp_buffer):
        self.model_network = model_network
        self.model_target_network = model_target_network
        self.exp_buffer = exp_buffer
        self.optimizer = optim.Adam(self.model_network.parameters(), lr=LR)
        self.loss = None
        self.criterion = nn.MSELoss()

    def update_model_network(self):
        self.optimizer.zero_grad()
        if len(self.exp_buffer) < BATCH_SIZE:
            batch = self.exp_buffer.sample(len(self.exp_buffer))
        else:
            batch = self.exp_buffer.sample(BATCH_SIZE)
        states, actions, rewards, dones, next_states = batch

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.ByteTensor(dones)

        state_action_values = self.model_network(states).max(1)[0]
        next_state_values = self.model_target_network(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + rewards
        self.loss = self.criterion(state_action_values, expected_state_action_values)
        self.loss.backward()
        self.optimizer.step()


class Agent:

    def __init__(self, exp_buffer):
        self.exp_buffer = exp_buffer
        self.env = GameAI()

    def get_state(self):
        # head = self.env.head

        # point_l = Point(head.x - SIZE, head.y)
        # point_r = Point(head.x + SIZE, head.y)
        # point_u = Point(head.x, head.y - SIZE)
        # point_d = Point(head.x, head.y + SIZE)
        
        # dir_l = self.env.snake.direction == Direction.LEFT
        # dir_r = self.env.snake.direction == Direction.RIGHT
        # dir_u = self.env.snake.direction == Direction.UP
        # dir_d = self.env.snake.direction == Direction.DOWN

        # danger_s = (dir_r and self.env.is_collision(point_r)) or (dir_l and self.env.is_collision(point_l)) or (dir_u and self.env.is_collision(point_u)) or (dir_d and self.env.is_collision(point_d))
        # danger_r = (dir_u and self.env.is_collision(point_r)) or (dir_d and self.env.is_collision(point_l)) or (dir_l and self.env.is_collision(point_u)) or (dir_r and self.env.is_collision(point_d))
        # danger_l = (dir_d and self.env.is_collision(point_r)) or (dir_u and self.env.is_collision(point_l)) or (dir_r and self.env.is_collision(point_u)) or (dir_l and self.env.is_collision(point_d))

        # food_l = self.env.food.x < self.env.head.x  # food left
        # food_r = self.env.food.x > self.env.head.x  # food right
        # food_u = self.env.food.y < self.env.head.y  # food up
        # food_d = self.env.food.y > self.env.head.y  # food down

        # state = [
        #     #direction
        #     dir_l,
        #     dir_r,
        #     dir_u,
        #     dir_d,

        #     # Food location 
        #     food_l,
        #     food_r,
        #     food_u,
        #     food_d,

        #     # Possible direction
        #     danger_l,
        #     danger_r,
        #     danger_s
        #     ]

        # # print("state: ", state)
        # return np.array(state, dtype=int)
        return self.env.state_grid()

    def get_action(self, model_network, state, epsilon):
        # Espilon-Greedy: tradeoff exploration / exploitation
        final_move = [0,0,0]
        if np.random.random() < epsilon:
            move = np.random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = model_network(state0)
            # print("prediction: ", prediction)
            move = torch.argmax(prediction).item()
            # print("move: ", move)
            # print("book:", torch.max(prediction).item())
            final_move[move] = 1

        return final_move

    def play_step(self, model_network, epsilon):
        self.env.reset()
        while True:

            # 1. Collect experience
            current_state = self.get_state()
            action = self.get_action(model_network, current_state, epsilon)
            reward, done, score = self.env.play(action)
            next_state = self.get_state()

            exp = Experience(current_state, action, reward, done, next_state)
            self.exp_buffer._append(exp)

            if done:
                return score

class Training:
    def __init__(self):
        self.exp_buffer = ExperienceBuffer(HISTORY_SIZE)
        self.agent = Agent(self.exp_buffer)
        self.model_network = DQN(128, 256, 3) #128, 512, 3
        self.model_target_network = DQN(128, 256, 3) #128, 512, 3
        self.model_trainer = DQN_trainer(self.model_network, self.model_target_network, self.exp_buffer)
        self.plotC = Plot()

        self.epoch = 0 
        self.best_score = 0

        # self.load()

    def train(self):
        while True:
            self.epoch += 1
            if self.agent.env.is_epsilon:
                epsilon = max(EPSILON_END, EPSILON_START - self.epoch * EPSILON_DECAY)
            else:
                epsilon = 0
            score = self.agent.play_step(self.model_network, epsilon)
            self.model_trainer.update_model_network()

            if self.agent.env.plot:
                self.plotC.update_lists(score, self.epoch)
            
            print('Game', self.epoch, 'Score', score, 'Record:', self.best_score, 'Epsilon:', epsilon)

            if score > self.best_score:
                self.best_score = score
                self.save()
            
            if self.epoch % SYNC_TARGET_EPOCH == 0:
                self.model_target_network.load_state_dict(self.model_network.state_dict())

            

    def save(self):
        torch.save({
            'epoch': self.epoch,
            'best_score': self.best_score,
            'model_network_state_dict': self.model_network.state_dict(),
            'model_target_network_state_dict': self.model_target_network.state_dict(),
            'optimizer_state_dict': self.model_trainer.optimizer.state_dict(),
            'loss': self.model_trainer.loss,
            'exp_buffer': self.exp_buffer,
            'total_score': self.plotC.total_score,
            'list_scores': self.plotC.list_scores,
            'list_mean_scores': self.plotC.list_mean_scores,
            'list_mean_10_scores': self.plotC.list_mean_10_scores,
            'timer': self.agent.env.time + self.agent.env.saved_time
        }, 'model/model.pth')

    def load(self):
        checkpoint = torch.load('model/model.pth')

        self.epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']

        self.model_network.load_state_dict(checkpoint['model_network_state_dict'])
        self.model_target_network.load_state_dict(checkpoint['model_target_network_state_dict'])
        self.model_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model_trainer.loss = checkpoint['loss']
        self.exp_buffer = checkpoint['exp_buffer']

        self.plotC.epoch = checkpoint['epoch']
        self.plotC.total_score = checkpoint['total_score']
        self.plotC.list_scores = checkpoint['list_scores']
        self.plotC.list_mean_scores = checkpoint['list_mean_scores']
        self.plotC.list_mean_10_scores = checkpoint['list_mean_10_scores']

        self.agent.env.saved_time = checkpoint['timer']

        self.model_network.eval()
        self.model_target_network.eval()

if __name__ == '__main__':
    training = Training()
    training.train() 