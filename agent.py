import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.distributions as distr
import numpy as np
import time

from collections import namedtuple, deque

from environments import Environment, Direction, Coordinates, Size_grid
from helper import Graphics

# Hyperparameters
BATCH_SIZE = 128
MEMORY_SIZE = 200

LR = 0.01 #0.001 #0.01
GAMMA = 0.95 #0.95 #0.9
CLIP_GRAD = 0.1
ENTROPY_BETA = 0.01

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))
Game_data = namedtuple('Game_data', ('idx_env', 'done', 'snake_coordinates', 'apple_coordinate', 'score', 'best_score', 'nbr_games'))

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
            nn.Softmax()
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
        
class A3C_trainer:
    def __init__(self, model_network, exp_buffer, epoch, loss_actor, loss_critic, best_score, end_process, speed):
        # shared networks
        self.model_network = model_network

        # shared variables
        ## mp.Queue
        self.exp_buffer = exp_buffer
        ## mp.Value
        self.epoch = epoch
        self.loss_actor_value = loss_actor
        self.loss_critic_value = loss_critic
        self.best_score = best_score
        self.end_process = end_process
        self.speed = speed
        # local variables
        self.optimizer = optim.Adam(self.model_network.parameters(), lr=LR, eps=1e-3)
        self.criterion = nn.MSELoss()
        self.exp_queue = deque(maxlen=MEMORY_SIZE)
        self.entropies = 0

        #load models
        # self.load_model()

        # Loop
        self.train_step()

    def train_step(self):
        current_best_score = 0
        
        while True:
            # Fill the experience memory
            self.fillin_exp_memory()

            # Update the model network
            self.update_model_network()

            # Update global variables
            ## epoch
            self.epoch.value += 1
            ## best score
            if self.best_score.value > current_best_score:
                current_best_score = self.best_score.value
                self.save_model()

            # End the process
            if self.end_process.value:
                break

            # Limitation speed
            if not self.speed.value:
                time.sleep(0.1)

    def update_model_network(self):
        # Pick first element of the experience memory
        game_exp = self.exp_queue.popleft()

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for element in game_exp:
            states.append(element.state)
            actions.append(element.action)
            rewards.append(element.reward)
            next_states.append(element.next_state)
            dones.append(element.done)

        # Convert to tensor
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        # print("states", states)
        # print("actions", actions)
        # print("rewards", rewards)
        # print("next_states", next_states)
        # print("dones", dones)

        # Compute the advantage
            # Compute the value of the current state
        probs, values = self.model_network(states)
            # Compute the value of the next state
        _, next_values = self.model_network(next_states)

        # print("probs", probs)
        # print("values", values.squeeze())
        # print("next_values", next_values.squeeze())

            # Compute the advantage
        advantages = rewards + GAMMA * next_values.squeeze() * (1 - dones) - values.squeeze()
        distribution = distr.Categorical(probs)
        log_probs = distribution.log_prob(actions)
        entropies = distribution.entropy()


        # print("advantages", advantages)
        # print("log_probs", log_probs)
        # print("entropies", entropies)

            # Compute the loss
        actor_loss = (-log_probs * advantages).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = ENTROPY_BETA * entropies.mean()
        loss = actor_loss + critic_loss + entropy_loss

        # print("actor_loss", actor_loss)
        # print("critic_loss", critic_loss)
        # print("entropy_loss", entropy_loss)
        # print("loss", loss)

        self.loss_actor_value.value = actor_loss.item()
        self.loss_critic_value.value = critic_loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fillin_exp_memory(self):
        while not self.exp_buffer.empty() or not self.exp_queue:
            self.exp_queue.append(self.exp_buffer.get())
            print("self.exp_queue", len(self.exp_queue))


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
        state_ts = torch.from_numpy(state).float()
        action_probs, _ = self.model_network(state_ts)
        distribution = distr.Categorical(probs = action_probs)
        move = distribution.sample()

        # Return it
        final_move = [0,0,0]
        final_move[move] = 1
        return final_move

    # Play in the game and give feedback to the NN every MAX_ITER_PER_STEP or when the game is over
    # For the training, trainer for NN need sequence of experience (MAX = MAX_ITER_PER_STEP)
    # However, sending all data will not be efficient.
    # So we will send the first state, the first action, the discounted reward and the last state.
    # If the game is over, we will also send the total undiscounted reward
    def play_step(self):
        game_exp = []
        while True:
            state = self.get_state()

            move = self.get_action(state)

            reward, done, score = self.env.play(move)

            # Get the new state
            new_state = self.get_state()

            # Add exp to list
            exp = Experience(state, np.argmax(move), reward, new_state, done)
            game_exp.append(exp)

            # Game data buffer (for display)
            data = Game_data(self.index, done, self.env.snake.snake_coordinates, self.env.apple.apple_coordinate, score, self.best_score, self.game_nbr)
            self.game_data_buffer.put(data)

            # 3. Restart game
            if done:
                
                self.exp_buffer.put(game_exp)

                self.env.reset()
                self.game_nbr += 1
                if score > self.best_score:
                    self.best_score = score

                game_exp = []

            # game speed
            if not self.speed.value:
                time.sleep(0.2)

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
    loss_actor = mp.Value('d', 0)
    loss_critic = mp.Value('d', 0)

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
    p_trainer = mp.Process(target=A3C_trainer, args=(model_network, exp_buffer, epoch, loss_actor, loss_critic, best_score, end_process, speed))
    p_graphic = mp.Process(target=Graphics, args=(size_grid, game_data_buffer, best_score, epoch, start_time, speed, random_init_snake, loss_actor, loss_critic, end_process))
    p_trainer.start()
    p_graphic.start()
    processes.append(p_trainer)
    processes.append(p_graphic)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()