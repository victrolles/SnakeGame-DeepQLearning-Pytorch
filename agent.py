import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import time

from collections import deque, namedtuple

from environments import Environment, Direction, Coordinates, Size_grid
from helper import Graphics


HISTORY_SIZE = 100_000
BATCH_SIZE = 1000
BUFFER_SIZE = 1000

LR = 0.01 #0.001
GAMMA = 0.9 #0.95

EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 0.001 #0.00001

SYNC_TARGET_EPOCH = 100

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'done', 'next_state'))
Game_data = namedtuple('Game_data', ('idx_env', 'snake_coordinates', 'apple_coordinate', 'score', 'best_score', 'nbr_games'))

class ExperienceMemory:
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
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.fully_connected_layers(x)
        
# (target=DQN_trainer, args=(model_network, model_target_network, exp_buffer, epoch, best_score, espilon, time))
class DQN_trainer:
    def __init__(self, model_network, model_target_network, exp_buffer, epoch, epsilon):
        # shared networks
        self.model_network = model_network
        self.model_target_network = model_target_network

        # shared variables
        ## mp.Queue
        self.exp_buffer = exp_buffer
        ## mp.Value
        self.epoch = epoch
        self.epsilon = epsilon
        # local variables
        self.optimizer = optim.Adam(self.model_network.parameters(), lr=LR)
        self.loss = None
        self.criterion = nn.MSELoss()
        self.exp_memory = ExperienceMemory(HISTORY_SIZE)

        # Loop
        self.train_step()

    def train_step(self):
        while True:
            self.fillin_exp_memory()
            self.update_model_network()
            self.sync_target_network()
            self.epoch.value += 1
            self.epsilon.value = max(EPSILON_END, EPSILON_START - self.epoch.value * EPSILON_DECAY)
            # print('Epoch', self.epoch.value, 'Epsilon:', self.epsilon.value)
            time.sleep(0.1)

    def update_model_network(self):
        self.optimizer.zero_grad()
        if len(self.exp_memory) < BATCH_SIZE:
            batch = self.exp_memory.sample(len(self.exp_memory))
        else:
            batch = self.exp_memory.sample(BATCH_SIZE)
        states, actions, rewards, dones, next_states = batch
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.ByteTensor(dones)
        state_action_values = self.model_network(states).gather(1, torch.argmax(actions, dim=1).unsqueeze(1)).squeeze(1)
        next_state_values = self.model_target_network(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * GAMMA + rewards
        self.loss = self.criterion(state_action_values, expected_state_action_values)
        self.loss.backward()
        self.optimizer.step()

    def fillin_exp_memory(self):
        while not self.exp_buffer.empty():
            self.exp_memory._append(self.exp_buffer.get())

    def sync_target_network(self):
        if self.epoch.value % SYNC_TARGET_EPOCH == 0:
            self.model_target_network.load_state_dict(self.model_network.state_dict())

class Agent:

    def __init__(self, index, size_grid, model_network, exp_buffer, game_data_buffer, espilon, speed, random_init_snake):
        # constant variables
        self.index = index
        self.size_grid = size_grid

        # shared variables
        ## mp.queue
        self.exp_buffer = exp_buffer
        self.game_data_buffer = game_data_buffer

        ## model
        self.model_network = model_network

        ## values
        self.epsilon = espilon
        self.speed = speed
        self.random_init_snake = random_init_snake

        # local variables
        # env
        self.env = Environment(size_grid)
        # others
        self.best_score = 0
        self.game_nbr = 0

        # Loop
        self.play_step()

    def get_state(self):
        head = self.env.snake.snake_coordinates[0]

        point_l = Coordinates(head.x - 1, head.y)
        point_r = Coordinates(head.x + 1, head.y)
        point_u = Coordinates(head.x, head.y - 1)
        point_d = Coordinates(head.x, head.y + 1)
        
        dir_l = self.env.snake.direction == Direction.LEFT
        dir_r = self.env.snake.direction == Direction.RIGHT
        dir_u = self.env.snake.direction == Direction.UP
        dir_d = self.env.snake.direction == Direction.DOWN

        danger_s = (dir_r and self.env.is_collision(point_r)) or (dir_l and self.env.is_collision(point_l)) or (dir_u and self.env.is_collision(point_u)) or (dir_d and self.env.is_collision(point_d))
        danger_r = (dir_u and self.env.is_collision(point_r)) or (dir_d and self.env.is_collision(point_l)) or (dir_l and self.env.is_collision(point_u)) or (dir_r and self.env.is_collision(point_d))
        danger_l = (dir_d and self.env.is_collision(point_r)) or (dir_u and self.env.is_collision(point_l)) or (dir_r and self.env.is_collision(point_u)) or (dir_l and self.env.is_collision(point_d))

        food_l = self.env.apple.apple_coordinate.x < self.env.snake.snake_coordinates[0].x  # food left
        food_r = self.env.apple.apple_coordinate.x > self.env.snake.snake_coordinates[0].x  # food right
        food_u = self.env.apple.apple_coordinate.y < self.env.snake.snake_coordinates[0].y  # food up
        food_d = self.env.apple.apple_coordinate.y > self.env.snake.snake_coordinates[0].y  # food down

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
        # return self.env.state_grid()

    def get_action(self, state):
        # Espilon-Greedy: tradeoff exploration / exploitation
        final_move = [0,0,0]
        if np.random.random() < self.epsilon.value:
            move = np.random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model_network(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def play_step(self):
        self.env.reset()
        while True:

            # 1. Collect experience
            current_state = self.get_state()
            action = self.get_action(current_state)
            reward, done, score = self.env.play(action)
            next_state = self.get_state()

            exp = Experience(current_state, action, reward, done, next_state)
            self.exp_buffer.put(exp)

            data = Game_data(self.index, self.env.snake.snake_coordinates, self.env.apple.apple_coordinate, score, self.best_score, self.game_nbr)
            self.game_data_buffer.put(data)

            # game speed
            time.sleep(0.5)
            # if not self.speed.value:
            #     time.sleep(0.5)

            if done:
                self.env.reset()
                self.env.snake.random_init = self.random_init_snake.value
                self.game_nbr += 1
                if score > self.best_score:
                    self.best_score = score

# class Training:
#     def __init__(self):
#         self.size_grid = Size_grid(10, 10)

#         self.exp_buffer = ExperienceBuffer(HISTORY_SIZE)
#         self.agent = Agent(self.exp_buffer, self.size_grid)
#         self.model_network = DQN(11, 256, 3) #128, 512, 3
#         self.model_target_network = DQN(11, 256, 3) #128, 512, 3
#         # self.model_network.share_memory()
#         self.model_trainer = DQN_trainer(self.model_network, self.model_target_network, self.exp_buffer)

#         self.epoch = 0 
#         self.best_score = 0
#         self.epsilon = EPSILON_START
#         self.time = time.time()

#         self.graphics = Graphics(self.size_grid, self.agent.env, self.epsilon, self.best_score, self.epoch, self.time)

#         # self.load()

#     def train(self):
#         while True:
#             self.epoch += 1
#             self.epsilon = max(EPSILON_END, EPSILON_START - self.epoch * EPSILON_DECAY)
#             # print("---------------------")
#             # start = time.perf_counter()
#             score = self.agent.play_step(self.model_network, self.epsilon)
#             # end = time.perf_counter()
#             # print("Time agent : ", end - start)
#             self.model_trainer.update_model_network()
#             # print("Time model : ", time.perf_counter() - end)

#             # self.plotC.update_lists(score, self.epoch)
#             # self.graphics.update_graphics()
            
#             print('Game', self.epoch, 'Score', score, 'Record:', self.best_score, 'Epsilon:', self.epsilon)

#             if score > self.best_score:
#                 self.best_score = score
#                 # self.save()
            
#             if self.epoch % SYNC_TARGET_EPOCH == 0:
#                 self.model_target_network.load_state_dict(self.model_network.state_dict())

            

#     def save(self):
#         torch.save({
#             'epoch': self.epoch,
#             'best_score': self.best_score,
#             'model_network_state_dict': self.model_network.state_dict(),
#             'model_target_network_state_dict': self.model_target_network.state_dict(),
#             'optimizer_state_dict': self.model_trainer.optimizer.state_dict(),
#             'loss': self.model_trainer.loss,
#             'exp_buffer': self.exp_buffer,
#             # 'total_score': self.plotC.total_score,
#             # 'list_scores': self.plotC.list_scores,
#             # 'list_mean_scores': self.plotC.list_mean_scores,
#             # 'list_mean_10_scores': self.plotC.list_mean_10_scores,
#         }, 'model/model.pth')

#     def load(self):
#         checkpoint = torch.load('model/model.pth')

#         self.epoch = checkpoint['epoch']
#         self.best_score = checkpoint['best_score']

#         self.model_network.load_state_dict(checkpoint['model_network_state_dict'])
#         self.model_target_network.load_state_dict(checkpoint['model_target_network_state_dict'])
#         self.model_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.model_trainer.loss = checkpoint['loss']
#         self.exp_buffer = checkpoint['exp_buffer']

#         # self.plotC.epoch = checkpoint['epoch']
#         # self.plotC.total_score = checkpoint['total_score']
#         # self.plotC.list_scores = checkpoint['list_scores']
#         # self.plotC.list_mean_scores = checkpoint['list_mean_scores']
#         # self.plotC.list_mean_10_scores = checkpoint['list_mean_10_scores']

#         self.model_network.eval()
#         self.model_target_network.eval()

def main():
    size_grid = Size_grid(10, 10)

    model_network = DQN(11, 256, 3) #128, 512, 3
    model_target_network = DQN(11, 256, 3) #128, 512, 3
    model_network.share_memory()
    model_target_network.share_memory()

    exp_buffer = mp.Queue(maxsize=BUFFER_SIZE)
    game_data_buffer = mp.Queue(maxsize=BUFFER_SIZE)

    epsilon = mp.Value('d', EPSILON_START)

    epoch = mp.Value('i', 0)
    best_score = mp.Value('i', 0)
    time = mp.Value('d', 0)

    speed = mp.Value('b', 0)
    random_init_snake = mp.Value('b', 0)

    processes = []

    for i in range(4):
        p_env = mp.Process(target=Agent, args=(i, size_grid, model_network, exp_buffer, game_data_buffer, epsilon, speed, random_init_snake))
        p_env.start()
        processes.append(p_env)
    p_trainer = mp.Process(target=DQN_trainer, args=(model_network, model_target_network, exp_buffer, epoch, epsilon))
    p_graphic = mp.Process(target=Graphics, args=(size_grid, game_data_buffer, epsilon, best_score, epoch, time, speed, random_init_snake))
    p_trainer.start()
    p_graphic.start()
    processes.append(p_trainer)
    processes.append(p_graphic)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()