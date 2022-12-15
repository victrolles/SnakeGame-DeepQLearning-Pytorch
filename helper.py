import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from collections import namedtuple
import time

Size_screen = namedtuple('Size_screen', ['width', 'height'])
Size_grid = namedtuple('Size_grid', ['width', 'height'])
Coordinates = namedtuple('Coordinates', ['x', 'y'])
Game_data = namedtuple('Game_data', ('idx_env', 'snake_coordinates', 'apple_coordinate', 'score', 'best_score', 'nbr_games'))

class Plot:
    def __init__(self):
        plt.ion()

        # self.fig, (self.ax1, self.ax2) = plt.subplots(1,2, figsize=(14,6))
        self.fig, self.ax1 = plt.subplots(1, figsize=(14,6))
        self.fig.suptitle('Learning Curves : Training')
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        plt.ioff()
        plt.close()

        self.list_scores = []
        self.list_mean_scores = []
        self.list_mean_10_scores = []
        self.total_score = 0
        self.epoch = 0

    def start_plotting(self):
        plt.ion()

    def stop_plotting(self):
        plt.ioff()
        plt.close()

    def update_lists(self, scores, epoch):
        self.epoch = epoch
        self.total_score += scores

        self.list_scores.append(scores)
        self.list_mean_scores.append(sum(self.list_scores) / len(self.list_scores))
        self.list_mean_10_scores.append(np.mean(self.list_scores[-50:]))

        self.update_plot()

    # def update_plot(self, scores, mean_scores, mean_10_scores, train_loss):
    def update_plot(self):
        
        self.ax1.clear()
        self.ax1.set_title('Scores :')
        self.ax1.set_xlabel('Number of Games')
        self.ax1.set_ylabel('score')
        self.ax1.plot(self.list_scores)
        self.ax1.plot(self.list_mean_scores)
        self.ax1.plot(self.list_mean_10_scores)
        self.ax1.set_ylim(ymin=0)
        self.ax1.text(len(self.list_scores)-1, self.list_scores[-1], str(self.list_scores[-1]))
        self.ax1.text(len(self.list_mean_scores)-1, self.list_mean_scores[-1], str(self.list_mean_scores[-1]))
        self.ax1.text(len(self.list_mean_10_scores)-1, self.list_mean_10_scores[-1], str(self.list_mean_10_scores[-1]))
        self.ax1.legend(['score', 'mean_score', 'tendancy'])       
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

class Graphics:
    def __init__(self, size_grid, game_data_buffer, espilon, best_score, epoch, time, speed, random_init_snake):
        # constant variables
        self.size_screen = Size_screen(1000, 600)
        self.size_grid = size_grid

        # shared variables
        ## mp.Queue
        self.game_data_buffer = game_data_buffer
        ## mp.Value
        ### double
        self.espilon = espilon
        self.time = time
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
        self.game_data = [Game_data(0, [Coordinates(0,0)], Coordinates(0,0), 0, 0, 0), Game_data(1, [Coordinates(0,0)], Coordinates(0,0), 0, 0, 0), Game_data(2, [Coordinates(0,0)], Coordinates(0,0), 0, 0, 0), Game_data(3, [Coordinates(0,0)], Coordinates(0,0), 0, 0, 0)]   

        # tkinter
        self.root = tk.Tk()
        self.root.title("Snake")
        self.root.geometry(f"{self.size_screen.width}x{self.size_screen.height}")
        self.root.resizable(False, False)
        self.root.configure(bg="#0000CC")
        self.root.update()

        # Loop
        self.update_graphics()

    def update_graphics(self):
        while True:
            self.update_game_data()
            self.display_all_environments()
            time.sleep(0.1)

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
    