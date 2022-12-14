import numpy as np
from enum import Enum
from collections import namedtuple
import time
import multiprocessing as mp

Size_screen = namedtuple('Size_screen', ['width', 'height'])
Size_grid = namedtuple('Size_grid', ['width', 'height'])
Coordinates = namedtuple('Coordinates', ['x', 'y'])

BACKGROUND_COLOR = (0, 102, 0)
RED = (255, 0, 0)
PINK = (255, 0, 255)
DARK_PINK = (102, 0, 102)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 153)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class Environment:
    def __init__(self, size_grid):
        self.size_grid = size_grid
        self.reset()

    def reset(self):
        self.snake = Snake(self.size_grid)
        self.apple = Apple(self.snake.snake_coordinates, self.size_grid)
        self.score = 0
        self.interation = 0

    def run(self, shared_list):
        while True:
            self.play(1)
            for idx, snake_coordinate in enumerate(self.snake.snake_coordinates):
                shared_list[2*idx] = snake_coordinate.x
                shared_list[2*idx+1] = snake_coordinate.y
            shared_list[-3] = self.apple.apple_coordinate.x
            shared_list[-2] = self.apple.apple_coordinate.y
            shared_list[-1] = self.score
            time.sleep(8)

    def play(self, action):
        # update or reset variables
        self.interation += 1
        reward = 0
        done = False

        # move snake
        self.snake.move(action)

        # check if snake is closer to apple
        print("snake head: ", self.snake.snake_coordinates[0])
        # print("apple: ", self.apple.apple_coordinate)
        current_dist = np.sqrt((self.snake.snake_coordinates[0].x - self.apple.apple_coordinate.x)**2 + (self.snake.snake_coordinates[0].y - self.apple.apple_coordinate.y)**2)
        old_dist = np.sqrt((self.snake.snake_coordinates[1].x - self.apple.apple_coordinate.x)**2 + (self.snake.snake_coordinates[1].y - self.apple.apple_coordinate.y)**2) if self.snake.length > 1 else 0
        if current_dist < old_dist:
            reward = 1
        else:
            reward = -1

        # check if snake eats apple
        if self.snake.snake_coordinates[0] == self.apple.apple_coordinate:
            self.snake.grow()
            self.apple.move()
            self.score += 1
            reward = 10

        # check if snake collides with itself or with the wall
        if self.is_collision() or self.interation > 100*self.snake.length:
            done = True
            reward = -100
            return reward, done, self.score

        return reward, done, self.score

    def is_collision(self):
        # check if snake collides with itself
        for i in range(3, self.snake.length):
            if self.snake.snake_coordinates[0] == self.snake.snake_coordinates[i]:
                return True

        # check if snake collides with the wall
        if self.snake.snake_coordinates[0].x < 0 or self.snake.snake_coordinates[0].x >= self.size_grid.width:
            return True
        if self.snake.snake_coordinates[0].y < 0 or self.snake.snake_coordinates[0].y >= self.size_grid.height:
            return True

        return False

class Snake:
    def __init__(self, size_grid):
        self.size_grid = size_grid
        self.init()

    def init(self):
        self.length = 1
        self.direction = Direction(np.random.randint(1,5))
        self.snake_coordinates = [Coordinates(np.random.randint(1,self.size_grid.width-1),np.random.randint(1,self.size_grid.height-1))]
        self.old_tail_coordinate = self.snake_coordinates[-1]

    def move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        if self.direction == Direction.RIGHT:
            new_snake_coordinate = Coordinates(self.snake_coordinates[0].x + 1, self.snake_coordinates[0].y)
        elif self.direction == Direction.LEFT:
            new_snake_coordinate = Coordinates(self.snake_coordinates[0].x - 1, self.snake_coordinates[0].y)
        elif self.direction == Direction.UP:
            new_snake_coordinate = Coordinates(self.snake_coordinates[0].x, self.snake_coordinates[0].y - 1)
        elif self.direction == Direction.DOWN:
            new_snake_coordinate = Coordinates(self.snake_coordinates[0].x, self.snake_coordinates[0].y + 1)

        self.old_tail_coordinate = self.snake_coordinates[-1]
        self.snake_coordinates.insert(0, new_snake_coordinate)
        self.snake_coordinates.pop()

    def grow(self):
        self.snake_coordinates.append(self.old_tail_coordinate)
        self.length += 1

class Apple:
    def __init__(self, snake_coordinates, size_grid):
        self.size_grid = size_grid
        self.apple_coordinate = Coordinates(0, 0)
        self.snake_coordinates = snake_coordinates
        self.move()

    def move(self):
        while True:
            self.apple_coordinate = Coordinates(np.random.randint(self.size_grid.width), np.random.randint(self.size_grid.height))
            for snake_coordinate in self.snake_coordinates:
                if self.apple_coordinate == snake_coordinate:
                    continue
            break

class Graphics:
    def __init__(self, size_screen, size_grid, environments):
        pass

def display(shared_memories):
    print("hello")
    while True:
        for idx, elem in enumerate(shared_memories):
            print("idx display: ", idx)
            print("head coordinate display: ", elem[0], elem[1])
        time.sleep(8)

if __name__ == '__main__':
    size_screen = Size_screen(1920, 1080)
    size_grid = Size_grid(10,8)
    size_shared_memory = 3+2*size_grid.width*size_grid.height
    environments = []
    processes = []
    shared_memories = []

    for _ in range(4):
        env = Environment(size_grid)
        shared_memory = mp.Array('i', size_shared_memory)
        for i in range(size_shared_memory):
            shared_memory[i] = -2
        process = mp.Process(target=env.run, args=(shared_memory,))

        process.start()

        environments.append(env)
        shared_memories.append(shared_memory)
        processes.append(process)

        time.sleep(1)

    graphic_process = mp.Process(target=display, args=(shared_memories,))
    graphic_process.start()

    for process in processes:
        process.join()
    graphic_process.join()
