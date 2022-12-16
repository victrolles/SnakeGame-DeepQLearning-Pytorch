import tkinter as tk
import time

root = tk.Tk()
root.title("Tkinter Test")
root.geometry("500x500")

canvas = tk.Canvas(root, width=400, height=400, bg="#009900", highlightthickness=0)
canvas.place(x=50, y=0)

canvas.create_rectangle(0, 0, 100, 100, fill="#000000")
canvas.create_rectangle(100, 100, 200, 200, fill="#000000")
canvas.create_rectangle(20, 20, 80, 80, fill="#990000")

print("objects in canvas: ", canvas.find_all())
print("one object", canvas.find_enclosed(99, 99, 201, 201))
root.update()
time.sleep(2)
canvas.delete(canvas.find_enclosed(99, 99, 201, 201)[0])


root.mainloop()