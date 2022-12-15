import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from collections import namedtuple

Size_screen = namedtuple('Size_screen', ['width', 'height'])
Size_grid = namedtuple('Size_grid', ['width', 'height'])
Coordinates = namedtuple('Coordinates', ['x', 'y'])

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
    def __init__(self, size_grid, env, epsilon, best_score, epoch, time):
        self.size_screen = Size_screen(1000, 600)

        self.size_grid = size_grid
        self.env = env
        self.epsilon = epsilon
        self.best_score = best_score
        self.epoch = epoch
        self.time = time

        self.root = tk.Tk()
        self.root.title("Snake")
        self.root.geometry(f"{self.size_screen.width}x{self.size_screen.height}")
        self.root.resizable(False, False)
        self.root.configure(bg="#0000CC")
        self.root.update()

        size_canvas = Size_screen(250, 250)
        self.pixel_size = int(size_canvas.width / self.size_grid.width)
        self.size_canvas = Size_screen(self.size_grid.width*self.pixel_size, self.size_grid.height*self.pixel_size)

        
        

    # def display_all_environments(self, shared_memories):
    #     self.display_one_environment(shared_memories[0], Coordinates(20, 20))
    #     self.display_one_environment(shared_memories[1], Coordinates(270, 20))
    #     self.display_one_environment(shared_memories[2], Coordinates(20, 270))
    #     self.display_one_environment(shared_memories[3], Coordinates(270, 270))

    def update_graphics(self):
        self.display_one_environment(Coordinates(20, 20))


    def display_one_environment(self, position_canvas):

        self.display_score(position_canvas)

        canvas_grid = tk.Canvas(self.root, width=self.size_canvas.width, height=self.size_canvas.height, bg="#009900", highlightthickness=0)
        canvas_grid = self.draw_apple(canvas_grid)
        canvas_grid = self.draw_snake(canvas_grid)
        canvas_grid = self.draw_grid(canvas_grid)
        canvas_grid.place(x=position_canvas.x, y=position_canvas.y)

        self.root.update()

    def display_score(self, position_canvas):
        canvas_score = tk.Canvas(self.root, width=100, height=20, bg="#0000CC", highlightthickness=0)
        canvas_score.create_text(50, 10, text=f"Score: {self.env.score}", fill="#FFFFFF", font=("Arial", 15))
        canvas_score.place(x=position_canvas.x, y=position_canvas.y-20)


    def draw_grid(self, canvas_grid):
        for i in range(0,self.size_grid.width+1):
            canvas_grid.create_line(i*self.pixel_size, 0, i*self.pixel_size, self.size_canvas.height, fill="#FFFFFF", width=2)
            canvas_grid.create_line(0, i*self.pixel_size, self.size_canvas.width, i*self.pixel_size, fill="#FFFFFF", width=2)
        return canvas_grid

    def draw_snake(self, canvas_grid):
        for snake_coordinate in self.env.snake.snake_coordinates:
            canvas_grid.create_rectangle(snake_coordinate.x*self.pixel_size, snake_coordinate.y*self.pixel_size, (snake_coordinate.x+1)*self.pixel_size, (snake_coordinate.y+1)*self.pixel_size, outline="#000066", fill="#000066", width=0)
        return canvas_grid

    def draw_apple(self, canvas_grid):
        canvas_grid.create_rectangle(self.env.apple.apple_coordinate.x*self.pixel_size, self.env.apple.apple_coordinate.y*self.pixel_size, (self.env.apple.apple_coordinate.x+1)*self.pixel_size, (self.env.apple.apple_coordinate.y+1)*self.pixel_size, outline="#FF0000", fill="#FF0000", width=0)
        return canvas_grid
    