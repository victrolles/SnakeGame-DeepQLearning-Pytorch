import pygame
from timeit import default_timer as timer
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from numba import cuda

from pygame.locals import * # input

SIZE = 40
SPEED = 2
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BACKGROUND_COLOR = (110, 110, 5)
SIZE_SCREEN = (1040,800) #multiple de 40 obligatoire 1040 800

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class Game:
    def __init__(self):
        pygame.init()
        self.surface = pygame.display.set_mode(SIZE_SCREEN)
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        # self.time = pygame.time.Clock()
        self.surface.fill((255,255,255))
        self.time = 0
        self.saved_time = 0
        self.reset()

        if cuda.is_available():
            print("cuda is available")

    
        

    def run(self): 
        
        while True:

            game_over, score = self.play()

            if game_over:
                break

        print("game over : " + str(game_over) + ", score : " + str(score))
        pygame.quit()

    def play(self):

        # 1. collect user input
        for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_UP:
                        if self.direction != Direction.DOWN:
                            self.direction = Direction.UP

                    if event.key == K_DOWN:
                        if self.direction != Direction.UP:
                            self.direction = Direction.DOWN

                    if event.key == K_RIGHT:
                        if self.direction != Direction.LEFT:
                            self.direction = Direction.RIGHT

                    if event.key == K_LEFT:
                        if self.direction != Direction.RIGHT:
                            self.direction = Direction.LEFT

                    if event.key == K_ESCAPE:
                        pygame.quit()
                        quit()

                elif event.type == QUIT:
                    pygame.quit()
                    quit()
        
        # 2. move
        self.render_background()
        self.snake.move(self.direction)
        self.head = Point(self.snake.x[0],self.snake.y[0])
        self.apple.draw()
        self.display_score()

        # BFS
        print("------------------------------------------------------")
        start_time = timer()
        
        if self.direction != Direction.LEFT:
            right = self.DFS(Point(self.head.x + SIZE, self.head.y), BLUE, occurence_test=True)
            print("right : " + str(right))
        if self.direction != Direction.RIGHT:
            left = self.DFS(Point(self.head.x - SIZE, self.head.y), RED, occurence_test=True)
            print("left : " + str(left))
        if self.direction != Direction.UP:
            down = self.DFS(Point(self.head.x, self.head.y + SIZE), GREEN, occurence_test=True)
            print("down : " + str(down))
        if self.direction != Direction.DOWN:
            up = self.DFS(Point(self.head.x, self.head.y - SIZE), YELLOW, occurence_test=True)
            print("up : " + str(up))

        print("time : " + str(timer() - start_time))


        pygame.display.flip()

        # 3. check if game over
        game_over = False
        ##  3.1. snake colliding with apple
        if self.collision(self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y):
            self.apple.move(self.snake)
            self.food = Point(self.apple.x,self.apple.y)
            self.snake.increase_length()
            self.score +=1

        ##  3.2. snake colliding
        if self.is_collision(self.head):
            game_over = True
            return game_over, self.score

        ##  3.3. victory
        if self.snake.length == int((SIZE_SCREEN[0]*SIZE_SCREEN[1])/(SIZE*SIZE)):
            print("win")
            game_over = True
            return game_over, self.score

        self.clock.tick(SPEED)

        return game_over, self.score  

    def display_score(self):
        self.time = int((pygame.time.get_ticks())/1000)

        font = pygame.font.SysFont('arial',30)
        score = font.render(f"Score: {self.score}", True, (255,255,255))
        self.surface.blit(score,(800,10))
        score2 = font.render(f"Timer: {self.time} s", True, (255,255,255))
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
        for i in range(1,self.snake.length):
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
        self.head = Point(self.snake.x[0],self.snake.y[0])

        self.apple = Apple(self.surface)
        self.food = Point(self.apple.x,self.apple.y)
        self.apple.draw()

        self.score = 0

    def draw_square(self, color, pos):
        pygame.draw.rect(self.surface, color, pygame.Rect(pos[0], pos[1], SIZE, SIZE))

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
                # self.draw_square(color, (new_state.x, new_state.y))
                if not occurence_test or new_state not in list_states_Explored:
                    iter+=1
                    # if iter > (self.snake.length-1)**2:
                    #     return iter
                    list_states_in_queue.append(new_state)
                    if occurence_test:
                        list_states_Explored.append(new_state)
        return iter


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

    def move(self, direction):
        
        self.direction = direction

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

if __name__ == "__main__":
    game = Game()
    game.run()