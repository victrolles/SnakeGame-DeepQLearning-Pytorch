import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import time

from collections import deque, namedtuple

from environments import Environment, Direction, Coordinates, Size_grid
from helper import Graphics

# Hyperparameters
BATCH_SIZE = 128
MAX_ITER_PER_STEP = 10
LR = 0.01 #0.001 #0.01
GAMMA = 0.9 #0.95 #0.9



SYNC_TARGET_EPOCH = 100

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'done'))
ExperienceFirstLast = namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))
Game_data = namedtuple('Game_data', ('idx_env', 'done', 'snake_coordinates', 'apple_coordinate', 'score', 'best_score', 'nbr_games'))

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


class A3C(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(A3C, self).__init__()

        # actor network
        self.policy_nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )

        # critic network
        self.value_nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.policy_nn(x), self.value_nn(x)
        
class DQN_trainer:
    def __init__(self, model_network, exp_buffer, epoch, epsilon, loss, epsilon_0, best_score, end_process, speed):
        # shared networks
        self.model_network = model_network

        # shared variables
        ## mp.Queue
        self.exp_buffer = exp_buffer
        ## mp.Value
        self.epoch = epoch
        self.epsilon = epsilon
        self.epsilon_0 = epsilon_0
        self.loss_value = loss
        self.best_score = best_score
        self.end_process = end_process
        self.speed = speed
        # local variables
        self.optimizer = optim.Adam(self.model_network.parameters(), lr=LR)
        self.loss = None
        self.criterion = nn.MSELoss()
        self.exp_memory = ExperienceMemory(BATCH_SIZE)

        #load models
        # self.load_model()

        # Loop
        self.train_step()

    def train_step(self):
        current_best_score = 0
        while True:
            self.fillin_exp_memory()
            self.update_model_network()
            self.epoch.value += 1
            if self.epsilon_0.value:
                self.epsilon.value = 0
            else:
                self.epsilon.value = max(EPSILON_END, EPSILON_START - self.epoch.value * EPSILON_DECAY)
            if self.best_score.value > current_best_score:
                current_best_score = self.best_score.value
                self.save_model()
            if self.end_process.value:
                break
            if not self.speed.value:
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
        self.loss_value.value = self.loss.item()
        self.loss.backward()
        self.optimizer.step()

    def fillin_exp_memory(self):
        while not self.exp_buffer.empty():
            self.exp_memory._append(self.exp_buffer.get())

    def save_model(self):
        torch.save({
            'model_network_state_dict': self.model_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score.value,
        }, 'model.pth')

    def load_model(self):
        checkpoint = torch.load('model.pth')
        self.model_network.load_state_dict(checkpoint['model_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_score.value = checkpoint['best_score']

        self.model_network.eval()

class Agent:

    def __init__(self, index, size_grid, model_network, exp_buffer, game_data_buffer, speed, random_init_snake, end_process):
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
        self.speed = speed
        self.random_init_snake = random_init_snake
        self.end_process = end_process

        # local variables
        # env
        self.env = Environment(size_grid, self.random_init_snake)
        # others
        self.best_score = 0
        self.game_nbr = 0

        # Loop
        self.play_step()

    def get_state(self):
        # return self.env.get_state_grid()

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

        return np.array(state, dtype=int)

    # The NN return the probability of each moves
    # We select a random action depending on the probability of each actions

    def get_action(self, state):
        # Select an action
        state0 = torch.tensor(state, dtype=torch.float)
        action_probs, _ = self.model_network(state0)
        move = np.random.choice([0,1,2], p=action_probs)

        # Return it
        final_move = [0,0,0]
        final_move[move] = 1
        return final_move

    # Play in the game and give feedback to the NN every MAX_ITER_PER_STEP or when the game is over
    # For the training, trainer for NN need sequence of experience (MAX = MAX_ITER_PER_STEP)
    # However, sending all data will not be efficient.
    # So we will send the first state, the first action, the discounted reward and the last state

    def play_step(self):
        self.env.reset()
        list_exp = []
        while True:

            # 1. Collect experience
            current_state = self.get_state()
            action = self.get_action(current_state)
            reward, done, score = self.env.play(action)

            # 2. Add experience to buffers
            ## 2.1 Experience buffer
            exp = Experience(current_state, action, reward, done)
            list_exp.append(exp)
            if done or self.env.iteration % MAX_ITER_PER_STEP == 0:
                # computer the discounted reward
                total_reward = 0
                for iter in reversed(list_exp):
                    total_reward *= GAMMA
                    total_reward += iter.reward
                expFL = ExperienceFirstLast(list_exp[0].state, list_exp[0].action, total_reward, list_exp[-1].state)
                self.exp_buffer.put(expFL)


            ## 2.2 Game data buffer (for display)
            data = Game_data(self.index, done, self.env.snake.snake_coordinates, self.env.apple.apple_coordinate, score, self.best_score, self.game_nbr)
            self.game_data_buffer.put(data)

            # game speed
            if not self.speed.value:
                time.sleep(0.5)
            else:
                time.sleep(0.01)

            # 3. Restart game
            if done:
                self.env.reset()
                self.game_nbr += 1
                if score > self.best_score:
                    self.best_score = score

            # 4. End process
            if self.end_process.value:
                break 


def main():
    size_grid = Size_grid(10, 10)

    model_network = A3C(11, 256, 3) #400, 512, 3
    model_network.share_memory()

    exp_buffer = mp.Queue(maxsize=BATCH_SIZE)
    game_data_buffer = mp.Queue(maxsize=BATCH_SIZE)

    start_time = mp.Value('d', time.time())
    loss = mp.Value('d', 0)

    epoch = mp.Value('i', 0)
    best_score = mp.Value('i', 0)

    speed = mp.Value('b', 0)
    random_init_snake = mp.Value('b', 0)
    end_process = mp.Value('b', 0)

    processes = []

    for i in range(4):
        p_env = mp.Process(target=Agent, args=(i, size_grid, model_network, exp_buffer, game_data_buffer, speed, random_init_snake, end_process))
        p_env.start()
        processes.append(p_env)
    p_trainer = mp.Process(target=DQN_trainer, args=(model_network, exp_buffer, epoch, loss, best_score, end_process, speed))
    p_graphic = mp.Process(target=Graphics, args=(size_grid, game_data_buffer, best_score, epoch, start_time, speed, random_init_snake, loss, end_process))
    p_trainer.start()
    p_graphic.start()
    processes.append(p_trainer)
    processes.append(p_graphic)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()