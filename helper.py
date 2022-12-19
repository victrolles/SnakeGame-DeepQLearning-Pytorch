import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import namedtuple
import time
import copy

Size_screen = namedtuple('Size_screen', ['width', 'height'])
Size_grid = namedtuple('Size_grid', ['width', 'height'])
Coordinates = namedtuple('Coordinates', ['x', 'y'])
Game_data = namedtuple('Game_data', ('idx_env', 'done', 'snake_coordinates', 'apple_coordinate', 'score', 'best_score', 'nbr_games'))
Environment = namedtuple('Environment', ('canvas_env','label_score'))

class Plot:
    def __init__(self, game_data, root, loss_actor, loss_critic, epoch):
        self.game_data = game_data
        self.root = root
        self.loss_actor = loss_actor
        self.loss_critic = loss_critic
        self.epoch = epoch

        self.fig, (self.graph_scores, self.graph_loss) = plt.subplots(1, 2, figsize=(8, 4))
        self.fig.suptitle('Learning Curves : Training')
        self.graph_loss2 = self.graph_loss.twinx()

        self.list_scores = [[0], [0], [0], [0]]
        self.list_loss_actor = [0]
        self.list_loss_critic = [0]

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().place(x=630, y=155)

    def update_training_data(self):
        if self.loss_actor.value != self.list_loss_actor[-1]:
            self.list_loss_actor.append(self.loss_actor.value)

        if self.loss_critic.value != self.list_loss_critic[-1]:
            self.list_loss_critic.append(self.loss_critic.value)

        for i in range(4):
            if self.game_data[i].done:
                self.list_scores[i].append(self.game_data[i].score)
                self.game_data[i] = self.game_data[i]._replace(done=False)

    def update_plot(self):
        # start = time.perf_counter()

        self.update_training_data()
        # print("------------------------------------")
        # print("update resources : ", time.perf_counter() - start)
        # start = time.perf_counter()
        
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

        # print("update plot 1 : ", time.perf_counter() - start)
        # start = time.perf_counter()

        self.graph_loss.clear()
        self.graph_loss.set_title('Loss :')
        self.graph_loss.set_xlabel('Number of Epochs')

        
        self.graph_loss.plot(self.list_loss_actor, color='red')
        # self.graph_loss.set_ylabel('actor_loss', color='red')
        # self.graph_loss.set_ylim(ymin=0)
        self.graph_loss.text(len(self.list_loss_actor)-1, self.list_loss_actor[-1], str(int(self.list_loss_actor[-1])))

        self.graph_loss2.clear()
        self.graph_loss2.plot(self.list_loss_critic, color='blue')
        # self.graph_loss2.set_ylabel('critic_loss', color='blue')
        # graph_loss2.set_ylim(ymin=0)
        self.graph_loss2.text(len(self.list_loss_critic)-1, self.list_loss_critic[-1], str(int(self.list_loss_critic[-1])))

        self.graph_loss.legend(['actor_loss'], loc='upper left')
        self.graph_loss2.legend(['critic_loss'], loc='upper right')

        # print("update plot 2 : ", time.perf_counter() - start)
        # start = time.perf_counter()

        self.canvas.draw()
        self.canvas.flush_events()

        # print("update canvas : ", time.perf_counter() - start)
        

class Graphics:
    def __init__(self, size_grid, game_data_buffer, best_score, epoch, time, speed, random_init_snake, loss_actor, loss_critic, end_process):
        # constant variables
        self.size_screen = Size_screen(1500, 600)
        self.size_grid = size_grid

        # shared variables
        ## mp.Queue
        self.game_data_buffer = game_data_buffer
        ## mp.Value
        ### double
        self.time = time
        self.loss_actor = loss_actor
        self.loss_critic = loss_critic
        ### int
        self.best_score = best_score
        self.epoch = epoch
        ### bool
        self.speed = speed
        self.random_init_snake = random_init_snake
        self.end_process = end_process

        # local variables
        self.is_display_envs = True
        self.is_display_infos = True
        self.is_display_plots = True
        self.fps = 0

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
        self.plot = Plot(self.game_data, self.root, self.loss_actor, self.loss_critic, self.epoch)

        # Loop
        self.update_graphics()

    def init_canvas(self):
        # canvas
        ## environments:
        self.canvas_envs = []
        self.canvas_location = [Coordinates(30, 30), Coordinates(320, 30), Coordinates(30, 320), Coordinates(320, 320)]
        for i in range(4):
            canvas_env = tk.Canvas(self.root, width=self.size_canvas.width, height=self.size_canvas.height, bg="#009900", highlightthickness=0)
            label_score = tk.Label(self.root, text=f"Score: {self.game_data[i].score} Best score: {self.game_data[i].best_score} Nbr games: {self.game_data[i].nbr_games}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 12))
            canvas_env = self.draw_grid(canvas_env)
            canvas_env.place(x=self.canvas_location[i].x, y=self.canvas_location[i].y)
            label_score.place(x=self.canvas_location[i].x, y=self.canvas_location[i].y-30)
            self.canvas_envs.append(Environment(canvas_env,label_score))

        ## infos:
        self.label_fps = tk.Label(self.root, text=f"FPS: {self.fps}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_epoch = tk.Label(self.root, text=f"Epoch: {self.epoch.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_time = tk.Label(self.root, text=f"Time: {self.time.value:.3f}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_best_score = tk.Label(self.root, text=f"Best score: {self.best_score.value}", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))

        ## buttons:
        ### display:
        self.button_display_envs = tk.Button(self.root, text="Display envs", command=self.display_envs, bg="#500000", fg="#FFFFFF", font=("Arial", 15))
        self.button_display_infos = tk.Button(self.root, text="Display infos", command=self.display_infos, bg="#500000", fg="#FFFFFF", font=("Arial", 15))
        self.button_display_plots = tk.Button(self.root, text="Display graphs", command=self.display_graphs, bg="#500000", fg="#FFFFFF", font=("Arial", 15))
        ### options:
        self.button_random_init_snake = tk.Button(self.root, text="Random init snake", command=self.change_random_init_snake, bg="#500000", fg="#FFFFFF", font=("Arial", 15))
        self.button_speed = tk.Button(self.root, text="Speed", command=self.change_speed, bg="#500000", fg="#FFFFFF", font=("Arial", 15))
        ### labels:
        self.label_button_display_envs = tk.Label(self.root, text="On", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_button_display_infos = tk.Label(self.root, text="On", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_button_display_plots = tk.Label(self.root, text="On", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_button_random_init_snake = tk.Label(self.root, text="Off", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        self.label_button_speed = tk.Label(self.root, text="Off", bg="#0000CC", fg="#FFFFFF", font=("Arial", 15))
        ### exit:
        self.button_exit = tk.Button(self.root, text="Exit", command=self.exit, bg="#500000", fg="#FFFFFF", font=("Arial", 15))

        self.label_fps.place(x=700, y=5)
        self.label_epoch.place(x=700, y=35)
        self.label_time.place(x=700, y=65)
        self.label_best_score.place(x=700, y=95)

        self.button_display_envs.place(x=900, y=10)
        self.button_display_infos.place(x=900, y=55)
        self.button_display_plots.place(x=900, y=100)

        self.button_random_init_snake.place(x=1160, y=10)
        self.button_speed.place(x=1160, y=55)

        self.label_button_display_envs.place(x=1050, y=15)
        self.label_button_display_infos.place(x=1050, y=60)
        self.label_button_display_plots.place(x=1050, y=105)
        self.label_button_random_init_snake.place(x=1350, y=15)
        self.label_button_speed.place(x=1350, y=60)

        self.button_exit.place(x=1425, y=60)

    def display_envs(self):
        if self.is_display_envs:
            self.is_display_envs = False
            self.label_button_display_envs.config(text="Off")
        else:
            self.is_display_envs = True
            self.label_button_display_envs.config(text="On")

    def display_infos(self):
        if self.is_display_infos:
            self.is_display_infos = False
            self.label_button_display_infos.config(text="Off")
        else:
            self.is_display_infos = True
            self.label_button_display_infos.config(text="On")

    def display_graphs(self):
        if self.is_display_plots:
            self.is_display_plots = False
            self.label_button_display_plots.config(text="Off")
        else:
            self.is_display_plots = True
            self.label_button_display_plots.config(text="On")

    def change_random_init_snake(self):
        if self.random_init_snake.value:
            self.random_init_snake.value = False
            self.label_button_random_init_snake.config(text="Off")
        else:
            self.random_init_snake.value = True
            self.label_button_random_init_snake.config(text="On")

    def change_speed(self):
        if self.speed.value:
            self.speed.value = False
            self.label_button_speed.config(text="Off")
        else:
            self.speed.value = True
            self.label_button_speed.config(text="On")

    def exit(self):
        self.root.destroy()
        self.end_process.value = True

    def update_graphics(self):
        self.init_canvas()
        iter=0
        while True:
            start = time.perf_counter()
            self.update_game_data()

            if self.is_display_envs:
                self.display_all_environments()

            if self.is_display_infos:
                self.display_training_infos()

            if self.is_display_plots and iter%50==0:
                self.plot.update_plot()

            time.sleep(0.01)
            self.fps = int(1 / (time.perf_counter() - start))
            self.root.update()
            iter+=1
            if iter > 1000:
                iter = 0

            if self.end_process.value:
                break
            

    def update_game_data(self):
        # get all data from the buffer
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
            # change best score
            if element.score > self.best_score.value:
                self.best_score.value = element.score

    def display_all_environments(self):
        for idx in range(4):
            self.display_environment(idx)

    def display_environment(self, idx):
        self.canvas_envs[idx].label_score.config(text=f"Score: {self.game_data[idx].score} Best score: {self.game_data[idx].best_score} Nbr games: {self.game_data[idx].nbr_games}")
        self.draw_apple(idx)
        self.draw_snake(idx)

    def display_training_infos(self):
        self.label_fps.config(text=f"FPS: {self.fps}")
        self.label_epoch.config(text=f"Epoch: {self.epoch.value}")
        self.label_time.config(text=f"Time: {(time.time() - self.time.value):.1f}")
        self.label_best_score.config(text=f"Best score: {self.best_score.value}")

    def draw_grid(self, canvas_env):
        for i in range(0,self.size_grid.width+1):
            canvas_env.create_line(i*self.pixel_size, 0, i*self.pixel_size, self.size_canvas.height, fill="#FFFFFF", width=2)
            canvas_env.create_line(0, i*self.pixel_size, self.size_canvas.width, i*self.pixel_size, fill="#FFFFFF", width=2)
        return canvas_env

    def draw_snake(self, idx):
        current_snake_index_rect = []
        for snake_coordinate in self.game_data[idx].snake_coordinates:

            top_left = Coordinates(snake_coordinate.x*self.pixel_size, snake_coordinate.y*self.pixel_size)
            bottom_right = Coordinates((snake_coordinate.x+1)*self.pixel_size, (snake_coordinate.y+1)*self.pixel_size)

            

            if self.canvas_envs[idx].canvas_env.find_enclosed(top_left.x-1, top_left.y-1, bottom_right.x+1, bottom_right.y+1):
                if self.canvas_envs[idx].canvas_env.find_enclosed(top_left.x-1, top_left.y-1, bottom_right.x+1, bottom_right.y+1)[0] in self.canvas_envs[idx].canvas_env.find_withtag("snake"):
                    pass
                else:
                    self.canvas_envs[idx].canvas_env.create_rectangle(top_left.x, top_left.y, bottom_right.x, bottom_right.y, outline="#000066", fill="#000066", width=0, tag="snake")
            else:
                self.canvas_envs[idx].canvas_env.create_rectangle(top_left.x, top_left.y, bottom_right.x, bottom_right.y, outline="#000066", fill="#000066", width=0, tag="snake")
            current_snake_index_rect.append(self.canvas_envs[idx].canvas_env.find_enclosed(top_left.x-1, top_left.y-1, bottom_right.x+1, bottom_right.y+1)[0])
        for idx_square in self.canvas_envs[idx].canvas_env.find_withtag("snake"):

            if idx_square not in current_snake_index_rect:
                self.canvas_envs[idx].canvas_env.delete(idx_square)

    def draw_apple(self, idx):
        if not self.canvas_envs[idx].canvas_env.find_withtag("apple"):
            self.canvas_envs[idx].canvas_env.create_rectangle(self.game_data[idx].apple_coordinate.x*self.pixel_size, self.game_data[idx].apple_coordinate.y*self.pixel_size, (self.game_data[idx].apple_coordinate.x+1)*self.pixel_size, (self.game_data[idx].apple_coordinate.y+1)*self.pixel_size, outline="#FF0000", fill="#FF0000", width=0, tag="apple")
        else:
            self.canvas_envs[idx].canvas_env.coords("apple", self.game_data[idx].apple_coordinate.x*self.pixel_size, self.game_data[idx].apple_coordinate.y*self.pixel_size, (self.game_data[idx].apple_coordinate.x+1)*self.pixel_size, (self.game_data[idx].apple_coordinate.y+1)*self.pixel_size)