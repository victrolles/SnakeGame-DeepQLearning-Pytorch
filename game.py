import pygame
import random
from enum import Enum
import numpy as np
from collections import namedtuple

from pygame.locals import * # input

pygame.init()

BACKGROUND_COLOR = (0, 102, 0)
RED = (255, 0, 0)
PINK = (255, 0, 255)
DARK_PINK = (102, 0, 102)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 153)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SIZE = 40
SPEED = 100
SIZE_SCREEN = (1040, 800) #multiple de 30

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
        self.time = 0
        self.saved_time = 0
        self.display = True
        self.random_init = False
        self.game_over = False
        self.reset()

    def run(self):
        # game loop
        while True:
            action = self.get_action()
            self.play(action)
            if self.game_over:
                self.display = False
                print("Score: ", self.score)
                print("Time: ", self.time)
                self.reset()
                break

    def play(self, action):
        # 1. collect user input
        for event in pygame.event.get():
                if event.type == KEYDOWN:

                    if event.key == K_ESCAPE:
                        pygame.quit()
                        quit()

                    if event.key == K_a:
                        if self.display:
                            self.display = False
                        else:
                            self.display = True

                    if event.key == K_r:
                        if self.random_init:
                            self.random_init = False
                        else:
                            self.random_init = True

                elif event.type == QUIT:
                    pygame.quit()
                    quit()

        # 2. move
        self.snake.move(action)    
        self.head = Point(self.snake.x[0],self.snake.y[0])
        if self.display:
            self.render_background()
            self.snake.draw()
            self.apple.draw()
            self.display_score()
            if not self.game_over:
                pygame.display.flip()

        # 3. check if game over
        self.game_over = False

        ##  3.1. snake colliding with apple
        if self.collision(self.snake.x[0], self.snake.y[0], self.apple.x, self.apple.y):
            self.apple.move(self.snake)
            self.food = Point(self.apple.x,self.apple.y)
            self.snake.increase_length()
            self.score +=1

        ##  3.2. snake colliding
        if self.is_collision(self.head):
            print("loose")
            self.game_over = True

        ##  3.4. victory
        if self.snake.length == int((SIZE_SCREEN[0]*SIZE_SCREEN[1])/(SIZE*SIZE)):
            print("win")
            self.game_over = True

        self.clock.tick(SPEED)

    def get_action(self):
        point_l = Point(self.head.x - SIZE, self.head.y)
        point_r = Point(self.head.x + SIZE, self.head.y)
        point_u = Point(self.head.x, self.head.y - SIZE)
        point_d = Point(self.head.x, self.head.y + SIZE)

        left = game.DFS(point_l, occurence_test=True)
        right = game.DFS(point_r, occurence_test=True)
        up = game.DFS(point_u, occurence_test=True)
        down = game.DFS(point_d, occurence_test=True)
        maxi = max(left, right, up, down)
        # print("left: ", left, "right: ", right, "up: ", up, "down: ", down)
        if 1 < left < maxi:
            print("max left", left)
        if 1 < right < maxi:
            print("max right", right)
        if 1 < up < maxi:
            print("max up", up)
        if 1 < down < maxi:
            print("max down", down)

        if left == maxi:
            left = 1
        else:
            left = 0
        if right == maxi:
            right = 1
        else:
            right = 0
        if up == maxi:
            up = 1
        else:
            up = 0
        if down == maxi:
            down = 1
        else:
            down = 0

        dist_l = 1313 - np.sqrt((point_l.x - self.food.x)**2 + (point_l.y - self.food.y)**2)
        dist_r = 1313 - np.sqrt((point_r.x - self.food.x)**2 + (point_r.y - self.food.y)**2)
        dist_u = 1313 - np.sqrt((point_u.x - self.food.x)**2 + (point_u.y - self.food.y)**2)
        dist_d = 1313 - np.sqrt((point_d.x - self.food.x)**2 + (point_d.y - self.food.y)**2)

        return np.argmax([left*dist_l, right*dist_r, up*dist_u, down*dist_d])

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
        self.surface.blit(score,(750,10))
        self.surface.blit(score2,(750,50))

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
        self.surface.fill(BACKGROUND_COLOR)

    def reset(self):
        self.snake = Snake(self.surface,self.random_init)
        if self.display:
            self.snake.draw()
        self.food = None
        self.apple = Apple(self.surface, self.snake)
        if self.display:
            self.apple.draw()
        
        self.score = 0
        self.head = Point(self.snake.x[0],self.snake.y[0])
        self.food = Point(self.apple.x,self.apple.y)
        self.frame_iteration = 0
        # self.game_over = False
        

    def next_state(self, state):
        list = []
        tempList = []

        tempList.append(Point(state.x + SIZE, state.y))
        tempList.append(Point(state.x - SIZE, state.y))
        tempList.append(Point(state.x, state.y + SIZE))
        tempList.append(Point(state.x, state.y - SIZE))

        for tempState in tempList:
            if not self.is_collision(tempState):
                available = True
                for i in range(self.snake.length):
                    if tempState.x == self.snake.x[i] and tempState.y == self.snake.y[i]:
                        available = False
                if available:
                    list.append(tempState)

        return list

    def DFS(self, initial_state, occurence_test=True):

        list_states_in_queue=[initial_state]
        list_states_Explored=[]
        iter=0
        for i in range(0,self.snake.length):
            if initial_state.x == self.snake.x[i] and initial_state.y == self.snake.y[i]:
                return 0

        while list_states_in_queue:
            iter+=1
            if iter > 200:
                return iter
            current_state=list_states_in_queue.pop(0)
            list_new_states=self.next_state(current_state)
            for new_state in list_new_states:
                if not occurence_test or new_state not in list_states_Explored:
                    
                    list_states_in_queue.append(new_state)
                    if occurence_test:
                        list_states_Explored.append(new_state)
        return iter

class Snake:
    def __init__(self, parent_screen, random_init=False):
        self.parent_screen = parent_screen
        if random_init:
            self.length = random.randint(10, 40)
            self.direction = Direction(random.randint(1,4))
            self.create_random_snake()
        else:
            self.length = 2
            self.direction = Direction.DOWN
            self.x = [SIZE_SCREEN[0] / 2, SIZE_SCREEN[0] / 2]
            self.y = [SIZE_SCREEN[1] / 2, SIZE_SCREEN[1] / 2 - SIZE]
        

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

    def draw(self):
        pygame.draw.rect(self.parent_screen, DARK_BLUE, (self.x[0], self.y[0], SIZE, SIZE))
        for i in range(1,self.length):
            pygame.draw.rect(self.parent_screen, BLUE, (self.x[i], self.y[i], SIZE, SIZE))

    def move(self, action):
        # [left, right, up, down]

        for i in range(self.length-1,0,-1):
            self.x[i] = self.x[i-1]
            self.y[i] = self.y[i-1]

        if action == 0:
            self.x[0] -= SIZE
            self.direction = Direction.LEFT
            # print("left")
        elif action == 1:
            self.x[0] += SIZE
            self.direction = Direction.RIGHT
            # print("right")
        elif action == 2:
            self.y[0] -= SIZE
            self.direction = Direction.UP
            # print("up")
        elif action == 3:
            self.y[0] += SIZE
            self.direction = Direction.DOWN
            # print("down")

class Apple:
    def __init__(self, parent_screen, snake):
        self.parent_screen = parent_screen
        self.x = 0
        self.y = 0
        self.move(snake)

    def draw(self):
        pygame.draw.rect(self.parent_screen, RED, (self.x, self.y, SIZE, SIZE))

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
    game = GameAI()
    game.run()