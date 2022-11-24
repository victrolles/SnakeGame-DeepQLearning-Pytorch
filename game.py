import pygame
import random
from enum import Enum
import numpy as np
from collections import namedtuple

from datetime import datetime
from pygame.locals import * # input

pygame.init()

BACKGROUND_COLOR = (110, 110, 5)

SIZE = 40
SPEED = 40
SIZE_SCREEN = (1040, 800) #multiple de 40 obligatoire 1040 800  , 1280,960

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class GameAI:
    def __init__(self):
        

        self.surface = pygame.display.set_mode(SIZE_SCREEN)
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.surface.fill((255,255,255))
        self.time = 0
        self.saved_time = 0
        self.reset()

    def play(self, action):
        self.frame_iteration +=1

        # 1. collect user input
        for event in pygame.event.get():
                if event.type == KEYDOWN:

                    if event.key == K_ESCAPE:
                        pygame.quit()
                        quit()

                elif event.type == QUIT:
                    pygame.quit()
                    quit()

        # 2. move
        self.render_background()
        self.snake.move(action)
        self.head = Point(self.snake.x[0],self.snake.y[0])
        self.apple.draw()
        self.display_score()
        pygame.display.flip()

        # 3. check if game over
        game_over = False
        reward = 0
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
            reward = -10
            return reward, game_over, self.score

        ##  3.4. victory
        if self.snake.length == int((SIZE_SCREEN[0]*SIZE_SCREEN[1])/(SIZE*SIZE)):
            print("win")
            game_over = True
            reward = 10
            return reward, game_over, self.score

        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def display_score(self):
        self.time = int((pygame.time.get_ticks())/1000)
        new_time = self.time + self.saved_time
        h = 0
        m = 0
        s = 0

        font = pygame.font.SysFont('arial',30)
        if new_time < 60:
            s = new_time
            score2 = font.render(f"Timer: {s} s", True, (255,255,255))
        elif new_time < 3600:
            m = new_time // 60
            s = new_time % 60
            score2 = font.render(f"Timer: {m} min {s} s", True, (255,255,255))
        else:
            h = new_time // 3600
            m = (new_time % 3600) // 60
            s = (new_time % 3600) % 60
            score2 = font.render(f"Timer: {h} hours {m} min {s} s", True, (255,255,255))

        
        score = font.render(f"Score: {self.score}", True, (255,255,255))
        self.surface.blit(score,(800,10))
        self.surface.blit(score2,(800,50))

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

    def render_background(self):
        bg = pygame.image.load("resources/background.jpg")
        self.surface.blit(bg, (0,0))

    def reset(self):
        self.direction = Direction.DOWN
        self.snake = Snake(self.surface,2, self.direction)
        self.snake.draw()
        self.food = None
        self.apple = Apple(self.surface)
        self.apple.draw()
        
        self.score = 0
        self.head = Point(self.snake.x[0],self.snake.y[0])
        self.food = Point(self.apple.x,self.apple.y)
        self.frame_iteration = 0

    def next_state(self, state):
        list = []
        tempList = []

        tempList.append(Point(state.x + SIZE, state.y))
        tempList.append(Point(state.x - SIZE, state.y))
        tempList.append(Point(state.x, state.y + SIZE))
        tempList.append(Point(state.x, state.y - SIZE))

        for tempState in tempList:
            if not self.is_collision(tempState):
                if tempState != self.head:
                    list.append(tempState)

        return list

    def DFS(self, initial_state, color,occurence_test=True):

        list_states_in_queue=[initial_state]
        list_states_Explored=[]
        iter=0

        while list_states_in_queue:
            current_state=list_states_in_queue.pop(0)
            list_new_states=self.next_state(current_state)
            for new_state in list_new_states:
                self.draw_square(color, (new_state.x, new_state.y))
                if not occurence_test or new_state not in list_states_Explored:
                    iter+=1
                    if iter>10:
                        return iter
                    list_states_in_queue.insert(0,new_state)
                    if occurence_test:
                        list_states_Explored.append(new_state)

        return iter

    def draw_square(self, color, pos):
        pygame.draw.rect(self.surface, color, pygame.Rect(pos[0], pos[1], SIZE, SIZE))

class Snake:
    def __init__(self, parent_screen, length, direction):
        self.parent_screen = parent_screen
        self.length = length
        self.block = pygame.image.load("resources/block.jpg").convert()
        self.x = [SIZE_SCREEN[0] / 2]*length
        self.y = [SIZE_SCREEN[1] / 2]*length
        self.direction = direction

    def increase_length(self):
        self.length+=1
        self.x.append(-1)
        self.y.append(-1)

    def draw(self):
        for i in range(self.length):
            self.parent_screen.blit(self.block,(self.x[i],self.y[i]))

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
        self.draw()

class Apple:
    def __init__(self, parent_screen):
        self.image = pygame.image.load("resources/apple.jpg").convert()
        self.parent_screen = parent_screen
        self.x = random.randint(0,SIZE_SCREEN[0]/SIZE-1)*SIZE
        self.y = random.randint(0,SIZE_SCREEN[1]/SIZE-1)*SIZE

    def draw(self):
        self.parent_screen.blit(self.image,(self.x,self.y))

    def move(self,snake):
        available_place = False
        while not available_place:
            available_place = True
            self.x = random.randint(0,SIZE_SCREEN[0]/SIZE-1)*SIZE
            self.y = random.randint(0,SIZE_SCREEN[1]/SIZE-1)*SIZE
            for i in range(snake.length):
                if self.x == snake.x[i] and self.y == snake.y[i]:
                    available_place = False
        self.draw()