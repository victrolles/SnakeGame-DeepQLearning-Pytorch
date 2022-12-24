import numpy as np
from enum import Enum
from collections import namedtuple

Size_screen = namedtuple('Size_screen', ['width', 'height'])
Size_grid = namedtuple('Size_grid', ['width', 'height'])
Coordinates = namedtuple('Coordinates', ['x', 'y'])

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class Environment:
    def __init__(self, size_grid, random_init):
        self.size_grid = size_grid
        self.random_init = random_init

        self.reset()

    def reset(self):
        self.snake = Snake(self.size_grid, self.random_init)
        self.apple = Apple(self.snake.snake_coordinates, self.size_grid)
        self.score = 0
        self.iteration = 0

    def play(self, action):
        # update or reset variables
        self.iteration += 1
        reward = 0
        done = False

        # move snake
        self.snake.move(action)

        # check if snake is closer to apple
        current_dist = np.sqrt((self.snake.snake_coordinates[0].x - self.apple.apple_coordinate.x)**2 + (self.snake.snake_coordinates[0].y - self.apple.apple_coordinate.y)**2)
        if self.snake.length > 1:
            old_dist = np.sqrt((self.snake.snake_coordinates[1].x - self.apple.apple_coordinate.x)**2 + (self.snake.snake_coordinates[1].y - self.apple.apple_coordinate.y)**2)
        else:
            old_dist = np.sqrt((self.snake.old_tail_coordinate.x - self.apple.apple_coordinate.x)**2 + (self.snake.old_tail_coordinate.y - self.apple.apple_coordinate.y)**2)
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
        if self.is_collision(self.snake.snake_coordinates[0]) or self.iteration > 100*self.snake.length:
            done = True
            reward = -10
            return reward, done, self.score

        return reward, done, self.score

    def is_collision(self, snake_head):
        # check if snake collides with itself
        for i in range(3, self.snake.length):
            if snake_head == self.snake.snake_coordinates[i]:
                return True

        # check if snake collides with the wall
        if snake_head.x < 0 or snake_head.x >= self.size_grid.width:
            return True
        if snake_head.y < 0 or snake_head.y >= self.size_grid.height:
            return True

        return False

    def get_state_grid(self):
        state_grid_frame1 = np.zeros((self.size_grid.width, self.size_grid.height, 2), dtype=int)
        state_grid_frame2 = np.zeros((self.size_grid.width, self.size_grid.height, 2), dtype=int)

        for snake_coordinate in self.snake.snake_coordinates:
            if 0 <= snake_coordinate.x < self.size_grid.width and 0 <= snake_coordinate.y < self.size_grid.height:
                state_grid_frame1[snake_coordinate.y, snake_coordinate.x] = [1,0]
        for snake_coordinate in self.snake.snake_coordinates[1:]:
            if 0 <= snake_coordinate.x < self.size_grid.width and 0 <= snake_coordinate.y < self.size_grid.height:
                state_grid_frame2[snake_coordinate.y, snake_coordinate.x] = [1,0]
        if 0 <= self.snake.snake_coordinates[-1].x < self.size_grid.width and 0 <= self.snake.snake_coordinates[-1].y < self.size_grid.height:
            state_grid_frame2[self.snake.snake_coordinates[-1].y, self.snake.snake_coordinates[-1].x] = [1,0]
        
        state_grid_frame2[self.apple.apple_coordinate.y, self.apple.apple_coordinate.x] = [0,1]
        state_grid_frame1[self.apple.apple_coordinate.y, self.apple.apple_coordinate.x] = [0,1]
        # print("self.snake.snake_coordinates: ", self.snake.snake_coordinates)
        # print("self.apple.apple_coordinate: ", self.apple.apple_coordinate)
        # print("state_grid_frame1: ", state_grid_frame1)
        return np.concatenate((state_grid_frame1.reshape((200)), state_grid_frame2.reshape((200))))

class Snake:
    def __init__(self, size_grid, random_init):
        self.size_grid = size_grid
        self.random_init = random_init
        self.init()

    def init(self):
        self.direction = Direction(np.random.randint(1,5))
        self.snake_coordinates = [Coordinates(np.random.randint(1,self.size_grid.width-2),np.random.randint(1,self.size_grid.height-2))]

        if self.random_init.value:
            self.length = np.random.randint(25, 30)
            # average time 0.0002s
            self.create_random_snake()
        else:
            self.length = 1

        self.old_tail_coordinate = self.snake_coordinates[-1]

    def create_random_snake(self):
        # initialize head of the snake
        unwanted_cells = self.snake_coordinates.copy()

        # initialize 2nd element of the snake
        if self.direction == Direction.RIGHT:
            self.snake_coordinates.append(Coordinates(self.snake_coordinates[0].x - 1, self.snake_coordinates[0].y))
            unwanted_cells.append(self.snake_coordinates[-1])
            unwanted_cells.append(Coordinates(self.snake_coordinates[0].x + 1, self.snake_coordinates[0].y))
        elif self.direction == Direction.LEFT:
            self.snake_coordinates.append(Coordinates(self.snake_coordinates[0].x + 1, self.snake_coordinates[0].y))
            unwanted_cells.append(self.snake_coordinates[-1])
            unwanted_cells.append(Coordinates(self.snake_coordinates[0].x - 1, self.snake_coordinates[0].y))
        elif self.direction == Direction.UP:
            self.snake_coordinates.append(Coordinates(self.snake_coordinates[0].x, self.snake_coordinates[0].y + 1))
            unwanted_cells.append(self.snake_coordinates[-1])
            unwanted_cells.append(Coordinates(self.snake_coordinates[0].x, self.snake_coordinates[0].y - 1))
        else:
            self.snake_coordinates.append(Coordinates(self.snake_coordinates[0].x, self.snake_coordinates[0].y - 1))
            unwanted_cells.append(self.snake_coordinates[-1])
            unwanted_cells.append(Coordinates(self.snake_coordinates[0].x, self.snake_coordinates[0].y + 1))

        # initialize the rest of the snake
        succeeded = False
        while not succeeded:
            succeeded = True
            temp_snake_coordinates = self.snake_coordinates.copy()
            temp_unwanted_cells = unwanted_cells.copy()
            for _ in range(2, self.length):
                cells_around = self.get_cells_around(temp_snake_coordinates[-1])
                index_cell = [0,1,2,3]
                np.random.shuffle(index_cell)
                for j in index_cell:
                    cell = cells_around[j]
                    if cell not in unwanted_cells and cell.x >= 0 and cell.x < self.size_grid.width and cell.y >= 0 and cell.y < self.size_grid.height:
                        temp_snake_coordinates.append(cell)
                        temp_unwanted_cells.append(cell)
                        break
                    if j == 3:
                        succeeded = False
                        break
                if not succeeded:
                    break
            if succeeded:
                self.snake_coordinates = temp_snake_coordinates.copy()
                

    def get_cells_around(self, cell):
        cells_around = []
        cells_around.append(Coordinates(cell.x + 1, cell.y))
        cells_around.append(Coordinates(cell.x - 1, cell.y))
        cells_around.append(Coordinates(cell.x, cell.y + 1))
        cells_around.append(Coordinates(cell.x, cell.y - 1))

        return cells_around

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
        good = True
        while True:
            good = True
            self.apple_coordinate = Coordinates(np.random.randint(self.size_grid.width), np.random.randint(self.size_grid.height))
            for snake_coordinate in self.snake_coordinates:
                if self.apple_coordinate == snake_coordinate:
                    good = False
            if good:
                break