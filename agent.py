import torch
import numpy as np
from collections import deque, namedtuple
from game import GameAI, Direction, Point, SIZE
import torch
import torch.nn as nn
import torch.optim as optim
from helper import Plot

HISTORY_SIZE = 100_000
BATCH_SIZE = 1000
REPLAY_START_SIZE = 100_000

LR = 0.001
GAMMA = 0.9

EPSILON_START = 1
EPSILON_END = 0.05
EPSILON_DECAY = 0.001

SYNC_TARGET_EPOCH = 100

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
        head = self.env.head

        point_l = Point(head.x - SIZE, head.y)
        point_r = Point(head.x + SIZE, head.y)
        point_u = Point(head.x, head.y - SIZE)
        point_d = Point(head.x, head.y + SIZE)
        
        dir_l = self.env.snake.direction == Direction.LEFT
        dir_r = self.env.snake.direction == Direction.RIGHT
        dir_u = self.env.snake.direction == Direction.UP
        dir_d = self.env.snake.direction == Direction.DOWN

        danger_s = (dir_r and self.env.is_collision(point_r)) or (dir_l and self.env.is_collision(point_l)) or (dir_u and self.env.is_collision(point_u)) or (dir_d and self.env.is_collision(point_d))
        danger_r = (dir_u and self.env.is_collision(point_r)) or (dir_d and self.env.is_collision(point_l)) or (dir_l and self.env.is_collision(point_u)) or (dir_r and self.env.is_collision(point_d))
        danger_l = (dir_d and self.env.is_collision(point_r)) or (dir_u and self.env.is_collision(point_l)) or (dir_r and self.env.is_collision(point_u)) or (dir_l and self.env.is_collision(point_d))

        food_l = self.env.food.x < self.env.head.x  # food left
        food_r = self.env.food.x > self.env.head.x  # food right
        food_u = self.env.food.y < self.env.head.y  # food up
        food_d = self.env.food.y > self.env.head.y  # food down

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
        self.model_network = DQN(11, 256, 3)
        self.model_target_network = DQN(11, 256, 3)
        self.model_trainer = DQN_trainer(self.model_network, self.model_target_network, self.exp_buffer)
        self.plotC = Plot()

    def train(self):
        epoch = 0
        best_score = 0
        epsilon = EPSILON_START

        # information for plotting
        list_score = []

        while True:
            epoch += 1
            epsilon = max(EPSILON_END, EPSILON_START - epoch * EPSILON_DECAY)
            score = self.agent.play_step(self.model_network, epsilon)

            list_score.append(score)
            self.plotC.update_plot(list_score)
            
            print('Game', epoch, 'Score', score, 'Record:', best_score, 'Epsilon:', epsilon)

            if score > best_score:
                best_score = score

            if len(self.exp_buffer) < REPLAY_START_SIZE:
                continue
            if epoch % SYNC_TARGET_EPOCH == 0:
                self.model_target_network.load_state_dict(self.model_network.state_dict())

            self.model_trainer.update_model_network()

if __name__ == '__main__':
    training = Training()
    training.train() 

# class Training:
#     def __init__(self):
#         self.plot_scores = []
#         self.plot_mean_scores = []
#         self.mean_10_scores = 0
#         self.plot_mean_10_scores = []
#         self.plot_train_loss = []
#         self.total_score = 0
#         self.record = 0
#         self.agent = Agent()
#         self.game = GameAI()
#         self.plotC = Plot()
#         self.load_nn()

#     def train(self):
#         while True:


#             if done:
#                 # train long memory, plot result
#                 self.game.reset()
#                 self.agent.epoch += 1
#                 self.agent.train_long_memory()

#                 if score > self.record:
#                     self.record = score
#                     self.save()

#                 print('Game', self.agent.epoch, 'Score', score, 'Record:', self.record)

#                 # information for plotting
#                 self.plot_scores.append(score)

#                 self.total_score += score

#                 mean_score = self.total_score / self.agent.epoch
#                 mean_10_score = np.mean(self.plot_scores[-10:])
#                 self.mean_10_scores = np.ceil(mean_score)

#                 self.plot_mean_scores.append(mean_score)
#                 self.plot_mean_10_scores.append(mean_10_score)

#                 self.plot_train_loss.append(self.agent.trainer.loss.item())

#                 self.plotC.update_plot(self.plot_scores, self.plot_mean_scores, self.plot_mean_10_scores, self.plot_train_loss)

#     def save(self):
#         torch.save({
#             'epoch': self.agent.epoch,
#             'model_state_dict': self.agent.model.state_dict(),
#             'optimizer_state_dict': self.agent.trainer.optimizer.state_dict(),
#             'loss': self.agent.trainer.loss,
#             'plot_scores': self.plot_scores,
#             'plot_mean_scores': self.plot_mean_scores,
#             'plot_mean_10_scores': self.plot_mean_10_scores,
#             'plot_train_loss': self.plot_train_loss,
#             'record': self.record,
#             'total_score': self.total_score,
#             'timer': self.game.time + self.game.saved_time,
#         }, 'model/model.pth')

#     def load_nn(self):
#         checkpoint = torch.load('model/model.pth')
#         self.agent.epoch = checkpoint['epoch']
#         self.agent.model.load_state_dict(checkpoint['model_state_dict'])
#         self.agent.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.agent.trainer.loss = checkpoint['loss']
#         self.plot_scores = checkpoint['plot_scores']
#         self.plot_mean_scores = checkpoint['plot_mean_scores']
#         self.plot_mean_10_scores = checkpoint['plot_mean_10_scores']
#         self.plot_train_loss = checkpoint['plot_train_loss']
#         self.record = checkpoint['record']
#         self.total_score = checkpoint['total_score']
#         self.game.saved_time = checkpoint['timer']
#         self.agent.model.eval()