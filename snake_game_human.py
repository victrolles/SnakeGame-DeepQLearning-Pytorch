# import pygame
import random
from enum import Enum
import numpy as np
from collections import namedtuple
import multiprocessing as mp
import time

from datetime import datetime
# from pygame.locals import * # input

# pygame.init()

BACKGROUND_COLOR = (0, 102, 0)
RED = (255, 0, 0)
PINK = (255, 0, 255)
DARK_PINK = (102, 0, 102)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 153)

SIZE = 40
SPEED = 10
SIZE_SCREEN = (1040, 800) #multiple de 40 : (26, 20)(1040, 800)     (240, 240) (320, 320)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class GameAI:
    def __init__(self):
        

        # self.surface = pygame.display.set_mode(SIZE_SCREEN)
        # pygame.display.set_caption('Snake')
        # self.clock = pygame.time.Clock()
        self.time = 0
        self.saved_time = 0

        self.display = True
        self.random_init = True
        self.plot = False
        self.speed = False
        self.is_epsilon = True

        self.reset()

    def run(self):
        while True:
            self.play(1)
            time.sleep(8)

    def play(self, action):
        self.frame_iteration +=1

        # 1. collect user input
        # for event in pygame.event.get():
        #         if event.type == KEYDOWN:

        #             if event.key == K_ESCAPE:
        #                 pygame.quit()
        #                 quit()

        #             if event.key == K_d:
        #                 if self.display:
        #                     self.display = False
        #                 else:
        #                     self.display = True

        #             if event.key == K_r:
        #                 if self.random_init:
        #                     self.random_init = False
        #                 else:
        #                     self.random_init = True

        #             if event.key == K_p:
        #                 if self.plot:
        #                     self.plot = False
        #                 else:
        #                     self.plot = True

        #             if event.key == K_s:
        #                 if self.speed:
        #                     self.speed = False
        #                 else:
        #                     self.speed = True

        #             if event.key == K_e:
        #                 if self.is_epsilon:
        #                     self.is_epsilon = False
        #                 else:
        #                     self.is_epsilon = True

        #         elif event.type == QUIT:
        #             pygame.quit()
        #             quit()

        # 2. move
        self.snake.move(action)    
        self.head = Point(self.snake.x[0],self.snake.y[0])
        print("head", self.head)
        # if self.display:
        #     self.render_background()
        #     self.snake.draw()
        #     self.apple.draw()
            # self.display_score()
            # pygame.display.flip()

        # 3. check if game over
        game_over = False
        reward = 0

        ##  3.0. closer to apple
        dist_current = np.sqrt((self.snake.x[0] - self.food.x)**2 + (self.snake.y[0] - self.food.y)**2)
        dist_previous = np.sqrt((self.snake.x[1] - self.food.x)**2 + (self.snake.y[1] - self.food.y)**2)
        if dist_current < dist_previous:
            reward = 1
        else:
            reward = -1

        ##  3.1. snake colliding with apple
        if self.collision(self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y):
            self.apple.move(self.snake)
            self.food = Point(self.apple.x,self.apple.y)
            self.snake.increase_length()
            self.score +=1
            reward = 10

        ##  3.2. snake colliding
        if self.is_collision(self.head) or self.frame_iteration > 100*self.snake.length:
            game_over = True
            reward = -100
            return reward, game_over, self.score

        ##  3.4. victory
        if self.snake.length == int((SIZE_SCREEN[0]*SIZE_SCREEN[1])/(SIZE*SIZE)):
            print("win")
            game_over = True
            reward = 10
            return reward, game_over, self.score

        # if not self.speed:
        #     self.clock.tick(SPEED)

        return reward, game_over, self.score

    def state_grid(self):
        grid1 = np.zeros((SIZE_SCREEN[1]//SIZE,SIZE_SCREEN[0]//SIZE),dtype=int)
        for i in range(self.snake.length):
            if 0 <= int(self.snake.y[i]//SIZE) < int(SIZE_SCREEN[1]//SIZE) and 0 <= int(self.snake.x[i]//SIZE) < int(SIZE_SCREEN[0]//SIZE):
                grid1[int(self.snake.y[i]//SIZE),int(self.snake.x[i]//SIZE)] = 1
        grid1[int(self.apple.y//SIZE),int(self.apple.x//SIZE)] = 2
        flat_grid1 = grid1.flatten()

        grid2 = np.zeros((SIZE_SCREEN[1]//SIZE,SIZE_SCREEN[0]//SIZE),dtype=int)
        for i in range(1,self.snake.length):
            if 0 <= int(self.snake.y[i]//SIZE) < int(SIZE_SCREEN[1]//SIZE) and 0 <= int(self.snake.x[i]//SIZE) < int(SIZE_SCREEN[0]//SIZE):
                grid2[int(self.snake.y[i]//SIZE),int(self.snake.x[i]//SIZE)] = 1
        grid2[int(self.snake.old_y//SIZE),int(self.snake.old_x//SIZE)] = 1
        grid2[int(self.apple.y//SIZE),int(self.apple.x//SIZE)] = 2
        flat_grid2 = grid2.flatten()
        flat_grid = np.concatenate((flat_grid1,flat_grid2))
        return flat_grid

    # def display_score(self):
    #     self.time = int((pygame.time.get_ticks())/1000)
    #     new_time = self.time + self.saved_time
    #     h = 0
    #     m = 0
    #     s = 0

    #     font = pygame.font.SysFont('arial',30)
    #     if new_time < 60:
    #         s = new_time
    #         score2 = font.render(f"Timer: {s} s", True, (255,255,255))
    #     elif new_time < 3600:
    #         m = new_time // 60
    #         s = new_time % 60
    #         score2 = font.render(f"Timer: {m} min {s} s", True, (255,255,255))
    #     else:
    #         h = new_time // 3600
    #         m = (new_time % 3600) // 60
    #         s = (new_time % 3600) % 60
    #         score2 = font.render(f"Timer: {h} hours {m} min {s} s", True, (255,255,255))

        
    #     score = font.render(f"Score: {self.score}", True, (255,255,255))
    #     self.surface.blit(score,(750,10))
    #     self.surface.blit(score2,(750,50))

    def collision(self,x1, y1, x2, y2):
        if x1 >= x2 and x1 < x2 + SIZE:
            if y1 >= y2 and y1 < y2 + SIZE:
                return True

        return False

    def is_collision(self, head=None):
        if head is None:
            head = self.head
        ##  snake colliding with itself
        for i in range(3,self.snake.length):
            if self.collision(head.x, head.y, self.snake.x[i], self.snake.y[i]):
                return True

        ##  snake colliding with the boundries of the window
        if not (0 <= head.x < SIZE_SCREEN[0] and 0 <= head.y < SIZE_SCREEN[1]):
            return True

        return False

    # def render_background(self):
    #     self.surface.fill(BACKGROUND_COLOR)

    def reset(self):
        self.snake = Snake(self.random_init)
        # if self.display:
        #     self.snake.draw()
        self.food = None
        self.apple = Apple(self.snake)
        # if self.display:
        #     self.apple.draw()
        
        self.score = 0
        self.head = Point(self.snake.x[0],self.snake.y[0])
        self.food = Point(self.apple.x,self.apple.y)
        self.frame_iteration = 0

class Snake:
    def __init__(self, random_init=False):
        # self.parent_screen = parent_screen
        if random_init:
            self.length = random.randint(2, 3)
            self.direction = Direction(random.randint(1,4))
            self.create_random_snake()
        else:
            self.length = 2
            self.direction = Direction.DOWN
            self.x = [SIZE_SCREEN[0] / 2, SIZE_SCREEN[0] / 2]
            self.y = [SIZE_SCREEN[1] / 2, SIZE_SCREEN[1] / 2 - SIZE]
        self.old_x = self.x[self.length-1]
        self.old_y = self.y[self.length-1]
        

    def create_random_snake(self):
        x = random.randint(2, SIZE_SCREEN[0] / SIZE - 2) * SIZE
        y = random.randint(2, SIZE_SCREEN[1] / SIZE - 2) * SIZE
        self.x = [x]
        self.y = [y]

        direction = self.direction
        for i in range(1, self.length):
            if direction == Direction.RIGHT:
                x = x - SIZE
                if x < 0:
                    x = x + SIZE
                    y=y-SIZE
                    if y < 0:
                        y = y +2*SIZE
                        direction = Direction.UP
                    else:
                        direction = Direction.DOWN
                self.x.append(x)
                self.y.append(y)
            elif direction == Direction.LEFT:
                x = x + SIZE
                if x > SIZE_SCREEN[0]-SIZE:
                    x = x - SIZE
                    y=y+SIZE
                    if y < 0:
                        y = y -2*SIZE
                        direction = Direction.DOWN
                    else:
                        direction = Direction.UP
                self.x.append(x)
                self.y.append(y)
            elif direction == Direction.UP:
                y = y + SIZE
                if y > SIZE_SCREEN[1]-SIZE:
                    y = y - SIZE
                    x=x-SIZE
                    if x < 0:
                        x = x +2*SIZE
                        direction = Direction.LEFT
                    else:
                        direction = Direction.RIGHT
                self.x.append(x)
                self.y.append(y)
            elif direction == Direction.DOWN:
                y = y - SIZE
                if y < 0:
                    y = y + SIZE
                    x=x+SIZE
                    if x < 0:
                        x = x -2*SIZE
                        direction = Direction.RIGHT
                    else:
                        direction = Direction.LEFT
                self.x.append(x)
                self.y.append(y)

    def increase_length(self):
        self.length+=1
        self.x.append(-1)
        self.y.append(-1)

    # def draw(self):
    #     pygame.draw.rect(self.parent_screen, DARK_BLUE, (self.x[0], self.y[0], SIZE, SIZE))
    #     for i in range(1,self.length):
    #         pygame.draw.rect(self.parent_screen, BLUE, (self.x[i], self.y[i], SIZE, SIZE))

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

        self.old_x = self.x[self.length-1]
        self.old_y = self.y[self.length-1]

        for i in range(self.length-1,0,-1):
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]

        if self.direction == Direction.UP:
            self.y[0] -= SIZE
        if self.direction == Direction.DOWN:
            self.y[0] += SIZE
        if self.direction == Direction.RIGHT:
            self.x[0] += SIZE
        if self.direction == Direction.LEFT:
            self.x[0] -= SIZE

class Apple:
    def __init__(self, snake):
        # self.parent_screen = parent_screen
        self.x = 0
        self.y = 0
        self.move(snake)

    # def draw(self):
    #     pygame.draw.rect(self.parent_screen, RED, (self.x, self.y, SIZE, SIZE))

    def move(self,snake):
        while True:
            available = True
            self.x = random.randint(0,SIZE_SCREEN[0]/SIZE-1)*SIZE
            self.y = random.randint(0,SIZE_SCREEN[1]/SIZE-1)*SIZE
            for i in range(snake.length):
                if self.x == snake.x[i] and self.y == snake.y[i]:
                    available = False
            if available:
                break

if __name__ == '__main__':
    environments = []
    processes = []

    for i in range(4):
        env = GameAI()
        p = mp.Process(target=env.run)
        p.start()
        environments.append(env)
        processes.append(p)
        time.sleep(1)

    for process in processes:
        process.join()
        