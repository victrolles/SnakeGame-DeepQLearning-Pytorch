import matplotlib.pyplot as plt
import numpy as np

class Plot:
    def __init__(self):
        plt.ion()

        # self.fig, (self.ax1, self.ax2) = plt.subplots(1,2, figsize=(14,6))
        self.fig, self.ax1 = plt.subplots(1, figsize=(14,6))
        self.fig.suptitle('Learning Curves : Training')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.list_scores = []
        self.list_mean_scores = []
        self.list_mean_10_scores = []
        self.total_score = 0
        self.epoch = 0

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

# class Graphics:
#     def __init__(self):
#         self.size_grid = Size_grid(8, 8)
#         self.size_screen = Size_screen(1000, 600)
#         size_canvas = Size_screen(250, 250)
#         self.pixel_size = int(size_canvas.width / self.size_grid.width)
#         self.size_canvas = Size_screen(self.size_grid.width*self.pixel_size, self.size_grid.height*self.pixel_size)
#         self.root = tk.Tk()
#         self.root.title("Snake")
#         self.root.geometry(f"{self.size_screen.width}x{self.size_screen.height}")
#         self.root.resizable(False, False)
#         self.root.configure(bg="#0000CC")
#         self.root.update()

#     def display_all_environments(self, shared_memories):
#         self.display_one_environment(shared_memories[0], Coordinates(20, 20))
#         self.display_one_environment(shared_memories[1], Coordinates(270, 20))
#         self.display_one_environment(shared_memories[2], Coordinates(20, 270))
#         self.display_one_environment(shared_memories[3], Coordinates(270, 270))


#     def display_one_environment(self, shared_memory, position_canvas):

#         snake_coordinates, apple_coordinate, score = self.retrieve_data_from_shared_memory(shared_memory)

#         canvas_score = tk.Canvas(self.root, width=100, height=20, bg="#0000CC", highlightthickness=0)
#         canvas_score.create_text(50, 10, text=f"Score: {score}", fill="#FFFFFF", font=("Arial", 15))
#         canvas_score.place(x=position_canvas.x, y=position_canvas.y-20)

#         canvas_grid = tk.Canvas(self.root, width=self.size_canvas.width, height=self.size_canvas.height, bg="#009900", highlightthickness=0)
#         for snake_coordinate in snake_coordinates:
#             canvas_grid.create_rectangle(snake_coordinate.x*self.pixel_size, snake_coordinate.y*self.pixel_size, (snake_coordinate.x+1)*self.pixel_size, (snake_coordinate.y+1)*self.pixel_size, outline="#000066", fill="#000066", width=0)
#         canvas_grid.create_rectangle(apple_coordinate.x*self.pixel_size, apple_coordinate.y*self.pixel_size, (apple_coordinate.x+1)*self.pixel_size, (apple_coordinate.y+1)*self.pixel_size, outline="#FF0000", fill="#FF0000", width=0)
#         for i in range(0,self.size_grid.width+1):
#             canvas_grid.create_line(i*self.pixel_size, 0, i*self.pixel_size, self.size_canvas.height, fill="#FFFFFF", width=2)
#             canvas_grid.create_line(0, i*self.pixel_size, self.size_canvas.width, i*self.pixel_size, fill="#FFFFFF", width=2)
#         canvas_grid.place(x=position_canvas.x, y=position_canvas.y)

#         self.root.update()

#     def retrieve_data_from_shared_memory(self, shared_memory):
#         snake_coordinates = []
#         for i in range(0, len(shared_memory)-4, 2):
#             if shared_memory[i] != -2:
#                 snake_coordinates.append(Coordinates(shared_memory[i], shared_memory[i+1]))
#         apple_coordinate = Coordinates(shared_memory[-3], shared_memory[-2])
#         score = shared_memory[-1]
#         return snake_coordinates, apple_coordinate, score
    