import numpy as np
from enum import Enum
from collections import namedtuple
import time
import multiprocessing as mp
import tkinter as tk

Size_screen = namedtuple('Size_screen', ['width', 'height'])
Size_grid = namedtuple('Size_grid', ['width', 'height'])
Coordinates = namedtuple('Coordinates', ['x', 'y'])

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

def display_all_environments(shared_memories):
    size_grid = Size_grid(8, 8)
    size_screen = Size_screen(1000, 600)
    size_canvas = Size_screen(250, 250)
    pixel_size = int(size_canvas.width / size_grid.width)
    size_canvas = Size_screen(size_grid.width*pixel_size, size_grid.height*pixel_size)
    root = tk.Tk()
    root.title("Snake")
    root.geometry(f"{size_screen.width}x{size_screen.height}")
    root.resizable(False, False)
    root.configure(bg="#0000CC")
    root.update()
    while True:
        display_one_environment(root, shared_memories[0], Coordinates(20, 20), size_grid, size_canvas, pixel_size)
        display_one_environment(root, shared_memories[1], Coordinates(310, 20), size_grid, size_canvas, pixel_size)
        display_one_environment(root, shared_memories[2], Coordinates(20, 310), size_grid, size_canvas, pixel_size)
        display_one_environment(root, shared_memories[3], Coordinates(310, 310), size_grid, size_canvas, pixel_size)
        time.sleep(1)


def display_one_environment(root, shared_memory, position_canvas, size_grid, size_canvas, pixel_size):

    snake_coordinates, apple_coordinate, score = retrieve_data_from_shared_memory(shared_memory)

    canvas_score = tk.Canvas(root, width=100, height=20, bg="#0000CC", highlightthickness=0)
    canvas_score.create_text(50, 10, text=f"Score: {score}", fill="#FFFFFF", font=("Arial", 15))
    canvas_score.place(x=position_canvas.x, y=position_canvas.y-20)

    canvas_grid = tk.Canvas(root, width=size_canvas.width, height=size_canvas.height, bg="#009900", highlightthickness=0)
    for snake_coordinate in snake_coordinates:
        canvas_grid.create_rectangle(snake_coordinate.x*pixel_size, snake_coordinate.y*pixel_size, (snake_coordinate.x+1)*pixel_size, (snake_coordinate.y+1)*pixel_size, outline="#000066", fill="#000066", width=0)
    canvas_grid.create_rectangle(apple_coordinate.x*pixel_size, apple_coordinate.y*pixel_size, (apple_coordinate.x+1)*pixel_size, (apple_coordinate.y+1)*pixel_size, outline="#FF0000", fill="#FF0000", width=0)
    for i in range(0,size_grid.width+1):
        canvas_grid.create_line(i*pixel_size, 0, i*pixel_size, size_canvas.height, fill="#FFFFFF", width=2)
        canvas_grid.create_line(0, i*pixel_size, size_canvas.width, i*pixel_size, fill="#FFFFFF", width=2)
    canvas_grid.place(x=position_canvas.x, y=position_canvas.y)
    root.update()

def retrieve_data_from_shared_memory(shared_memory):
    snake_coordinates = []
    for i in range(0, len(shared_memory)-4, 2):
        if shared_memory[i] != -2:
            snake_coordinates.append(Coordinates(shared_memory[i], shared_memory[i+1]))
    apple_coordinate = Coordinates(shared_memory[-3], shared_memory[-2])
    score = shared_memory[-1]
    return snake_coordinates, apple_coordinate, score

def display(shared_memories):
    while True:
        for idx, shared_memory in enumerate(shared_memories):
            print(idx," : " ,shared_memory[0],shared_memory[1])
        time.sleep(1)

if __name__ == '__main__':
    size_screen = Size_screen(1920, 1080)
    size_grid = Size_grid(8,8)
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

    graphic_process = mp.Process(target=display_all_environments, args=(shared_memories,))
    graphic_process.start()

    for process in processes:
        process.join()
    graphic_process.join()
