import tkinter as tk
from collections import namedtuple
import time
import torch.multiprocessing as mp

Size_screen = namedtuple('Size_screen', ['width', 'height'])
Size_grid = namedtuple('Size_grid', ['width', 'height'])
Coordinates = namedtuple('Coordinates', ['x', 'y'])


class tkinter_test:
    def __init__(self, shared_memory, position_canvas):
        self.size_grid = Size_grid(8, 8)
        self.size_screen = Size_screen(1000, 600)
        size_canvas = Size_screen(250, 250)
        self.pixel_size = int(size_canvas.width / self.size_grid.width)
        self.size_canvas = Size_screen(self.size_grid.width*self.pixel_size, self.size_grid.height*self.pixel_size)
        self.root = tk.Tk()
        self.root.title("Snake")
        self.root.geometry(f"{self.size_screen.width}x{self.size_screen.height}")
        self.root.resizable(False, False)
        self.root.configure(bg="#0000CC")
        self.root.update()
        self.display_one_environment(shared_memory, position_canvas)

    # def display_all_environments(self, shared_memories):
    #     self.display_one_environment(shared_memories[0], Coordinates(20, 20))
    #     self.display_one_environment(shared_memories[1], Coordinates(270, 20))
    #     self.display_one_environment(shared_memories[2], Coordinates(20, 270))
    #     self.display_one_environment(shared_memories[3], Coordinates(270, 270))


    def display_one_environment(self, shared_memory, position_canvas):
        while True:
            data = shared_memory.get()
            snake_coordinates, apple_coordinate, score = self.retrieve_data_from_shared_memory(data)

            canvas_score = tk.Canvas(self.root, width=100, height=20, bg="#0000CC", highlightthickness=0)
            canvas_score.create_text(50, 10, text=f"Score: {score}", fill="#FFFFFF", font=("Arial", 15))
            canvas_score.place(x=position_canvas.x, y=position_canvas.y-20)

            canvas_grid = tk.Canvas(self.root, width=self.size_canvas.width, height=self.size_canvas.height, bg="#009900", highlightthickness=0)
            for snake_coordinate in snake_coordinates:
                canvas_grid.create_rectangle(snake_coordinate.x*self.pixel_size, snake_coordinate.y*self.pixel_size, (snake_coordinate.x+1)*self.pixel_size, (snake_coordinate.y+1)*self.pixel_size, outline="#000066", fill="#000066", width=0)
            canvas_grid.create_rectangle(apple_coordinate.x*self.pixel_size, apple_coordinate.y*self.pixel_size, (apple_coordinate.x+1)*self.pixel_size, (apple_coordinate.y+1)*self.pixel_size, outline="#FF0000", fill="#FF0000", width=0)
            for i in range(0,self.size_grid.width+1):
                canvas_grid.create_line(i*self.pixel_size, 0, i*self.pixel_size, self.size_canvas.height, fill="#FFFFFF", width=2)
                canvas_grid.create_line(0, i*self.pixel_size, self.size_canvas.width, i*self.pixel_size, fill="#FFFFFF", width=2)
            canvas_grid.place(x=position_canvas.x, y=position_canvas.y)

            self.root.update()

    def retrieve_data_from_shared_memory(self, shared_memory):
        snake_coordinates = []
        for i in range(0, len(shared_memory)-4, 2):
            if shared_memory[i] != -2:
                snake_coordinates.append(Coordinates(shared_memory[i], shared_memory[i+1]))
        apple_coordinate = Coordinates(shared_memory[-3], shared_memory[-2])
        score = shared_memory[-1]
        return snake_coordinates, apple_coordinate, score

def generate_shared_memory(env_data):
    while True:
        env_data.put((1,2,2,2,-2,-2,7,7,1))
        time.sleep(2)
        env_data.put((0,2,1,2,-2,-2,7,7,1))
        time.sleep(2)
        env_data.put((0,3,0,2,-2,-2,7,7,1))
        time.sleep(2)

if __name__ == '__main__':
    position_canvas = Coordinates(20, 20)
    env_data = mp.Queue()

    p_generator = mp.Process(target=generate_shared_memory, args=(env_data,))
    p_display = mp.Process(target=tkinter_test, args=(env_data,position_canvas))

    p_generator.start()
    p_display.start()

    p_generator.join()
    p_display.join()

    # tk_test.display_one_environment((1,2,2,2,-2,-2,7,7,1), position_canvas)
    # time.sleep(2)
    # tk_test.display_one_environment((0,2,1,2,-2,-2,7,7,1), position_canvas)
    # time.sleep(2)
    # tk_test.display_one_environment((0,3,0,2,-2,-2,7,7,1), position_canvas)
    # tk_test.root.mainloop()



