import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import namedtuple
import time

Size_screen = namedtuple('Size_screen', ['width', 'height'])
Size_grid = namedtuple('Size_grid', ['width', 'height'])
Coordinates = namedtuple('Coordinates', ['x', 'y'])
Game_data = namedtuple('Game_data', ('idx_env', 'done', 'snake_coordinates', 'apple_coordinate', 'score', 'best_score', 'nbr_games'))

class Plot:
    def __init__(self, game_data, root, loss, epoch):
        self.game_data = game_data
        self.root = root
        self.loss = loss
        self.epoch = epoch

        self.fig, (self.graph_scores, self.graph_loss) = plt.subplots(1, 2, figsize=(8, 4))
        self.fig.suptitle('Learning Curves : Training')

        self.list_scores = [[0], [0], [0], [0]]
        self.list_loss = [0]

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().place(x=630, y=155)

    def update_training_data(self):
        if self.loss.value != self.list_loss[-1]:
            self.list_loss.append(self.loss.value)

        for i in range(4):
            if self.game_data[i].done:
                self.list_scores[i].append(self.game_data[i].score)
                self.game_data[i] = self.game_data[i]._replace(done=False)

    def update_plot(self):
        self.update_training_data()
        
        self.graph_scores.clear()
        self.graph_scores.set_title('Scores :')
        self.graph_scores.set_xlabel('Number of Games')
        self.graph_scores.set_ylabel('score')
        self.graph_scores.plot(self.list_scores[0])
        self.graph_scores.plot(self.list_scores[1])
        self.graph_scores.plot(self.list_scores[2])
        self.graph_scores.plot(self.list_scores[3])
        self.graph_scores.set_ylim(ymin=0)
        self.graph_scores.text(len(self.list_scores[0])-1, self.list_scores[0][-1], str(self.list_scores[0][-1]))
        self.graph_scores.text(len(self.list_scores[1])-1, self.list_scores[1][-1], str(self.list_scores[1][-1]))
        self.graph_scores.text(len(self.list_scores[2])-1, self.list_scores[2][-1], str(self.list_scores[2][-1]))
        self.graph_scores.text(len(self.list_scores[3])-1, self.list_scores[3][-1], str(self.list_scores[3][-1]))
        self.graph_scores.legend(['env 0', 'env 1', 'env 2', 'env 3'])

        self.graph_loss.clear()
        self.graph_loss.set_title('Loss :')
        self.graph_loss.set_xlabel('Number of Epochs')
        self.graph_loss.set_ylabel('loss')
        self.graph_loss.plot(self.list_loss)
        self.graph_loss.set_ylim(ymin=0)
        self.graph_loss.text(len(self.list_loss)-1, self.list_loss[-1], str(self.list_loss[-1]))
        self.graph_loss.legend(['loss'])

        self.canvas.draw()

class Graphics:
    def __init__(self, size_grid, game_data_buffer, espilon, best_score, epoch, time, speed, random_init_snake, loss):
        # constant variables
        self.size_screen = Size_screen(1500, 600)
        self.size_grid = size_grid

        # shared variables
        ## mp.Queue
        self.game_data_buffer = game_data_buffer
        ## mp.Value
        ### double
        self.espilon = espilon
        self.time = time
        self.loss = loss
        ### int
        self.best_score = best_score
        self.epoch = epoch
        ### bool
        self.speed = speed
        self.random_init_snake = random_init_snake

        # local variables
        size_canvas = Size_screen(250, 250)
        self.pixel_size = int(size_canvas.width / self.size_grid.width)
        self.size_canvas = Size_screen(self.size_grid.width*self.pixel_size, self.size_grid.height*self.pixel_size)
        self.game_data = [Game_data(0, False, [Coordinates(0,0)], Coordinates(0,0), 0, 0, 0), Game_data(1, False, [Coordinates(0,0)], Coordinates(0,0), 0, 0, 0), Game_data(2, False, [Coordinates(0,0)], Coordinates(0,0), 0, 0, 0), Game_data(3, False, [Coordinates(0,0)], Coordinates(0,0), 0, 0, 0)]   

        # tkinter
        self.root = tk.Tk()
        self.root.title("Snake")
        self.root.geometry(f"{self.size_screen.width}x{self.size_screen.height}")
        self.root.resizable(False, False)
        self.root.configure(bg="#0000CC")
        self.root.update()

        # matplotlib
        self.plot = Plot(self.game_data, self.root, self.loss, self.epoch)

        # Loop
        self.update_graphics()

    def update_graphics(self):
        while True:
            self.update_game_data()
            self.display_all_environments()
            self.plot.update_plot()
            self.display_training_infos()
            time.sleep(0.03)

    def update_game_data(self):
        while not self.game_data_buffer.empty():
            element = self.game_data_buffer.get()
            if element.idx_env == 0:
                self.game_data[0] = element
            elif element.idx_env == 1:
                self.game_data[1] = element
            elif element.idx_env == 2:
                self.game_data[2] = element 
            elif element.idx_env == 3:
                self.game_data[3] = element

    def display_all_environments(self):
        self.display_one_environment(0, Coordinates(20, 20))
        self.display_one_environment(1, Coordinates(310, 20))
        self.display_one_environment(2, Coordinates(20, 310))
        self.display_one_environment(3, Coordinates(310, 310))

    def display_one_environment(self, idx, position_canvas):

        self.display_infos(idx, position_canvas)

        canvas_grid = tk.Canvas(self.root, width=self.size_canvas.width, height=self.size_canvas.height, bg="#009900", highlightthickness=0)
        canvas_grid = self.draw_apple(idx, canvas_grid)
        canvas_grid = self.draw_snake(idx, canvas_grid)
        canvas_grid = self.draw_grid(canvas_grid)
        canvas_grid.place(x=position_canvas.x, y=position_canvas.y)

        self.root.update()

    def display_infos(self, idx, position_canvas):
        canvas_score = tk.Canvas(self.root, width=250, height=20, bg="#0000CC", highlightthickness=0)
        canvas_score.create_text(125, 10, text=f"Score: {self.game_data[idx].score} Best score: {self.game_data[idx].best_score} Nbr games: {self.game_data[idx].nbr_games}", fill="#FFFFFF", font=("Arial", 11))
        canvas_score.place(x=position_canvas.x, y=position_canvas.y-20)

    def display_training_infos(self):
        canvas_training_infos = tk.Canvas(self.root, width=250, height=20, bg="#0000CC", highlightthickness=0)
        canvas_training_infos.create_text(125, 10, text=f"Epoch: {self.epoch.value}, Epsilon: {self.espilon.value:.3f}", fill="#FFFFFF", font=("Arial", 11))
        canvas_training_infos.place(x=755, y=30)

        canvas_training_infos = tk.Canvas(self.root, width=250, height=20, bg="#0000CC", highlightthickness=0)
        canvas_training_infos.create_text(125, 10, text=f"Time: {(time.time() - self.time.value):.1f}", fill="#FFFFFF", font=("Arial", 11))
        canvas_training_infos.place(x=755, y=60)

    def draw_grid(self, canvas_grid):
        for i in range(0,self.size_grid.width+1):
            canvas_grid.create_line(i*self.pixel_size, 0, i*self.pixel_size, self.size_canvas.height, fill="#FFFFFF", width=2)
            canvas_grid.create_line(0, i*self.pixel_size, self.size_canvas.width, i*self.pixel_size, fill="#FFFFFF", width=2)
        return canvas_grid

    def draw_snake(self, idx, canvas_grid):
        for snake_coordinate in self.game_data[idx].snake_coordinates:
            canvas_grid.create_rectangle(snake_coordinate.x*self.pixel_size, snake_coordinate.y*self.pixel_size, (snake_coordinate.x+1)*self.pixel_size, (snake_coordinate.y+1)*self.pixel_size, outline="#000066", fill="#000066", width=0)
        return canvas_grid

    def draw_apple(self, idx, canvas_grid):
        canvas_grid.create_rectangle(self.game_data[idx].apple_coordinate.x*self.pixel_size, self.game_data[idx].apple_coordinate.y*self.pixel_size, (self.game_data[idx].apple_coordinate.x+1)*self.pixel_size, (self.game_data[idx].apple_coordinate.y+1)*self.pixel_size, outline="#FF0000", fill="#FF0000", width=0)
        return canvas_grid