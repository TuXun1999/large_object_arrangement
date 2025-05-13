import tkinter as tk

window = tk.Tk()

window.columnconfigure([0, 1, 2, 3, 4, 5, 6], weight=1, minsize=75)
window.rowconfigure([0, 1], weight=1, minsize=50)

buttons = {}
buttons["w"] = [0, 1]
buttons["a"] = [1, 0]
buttons["s"] = [1, 1]
buttons["d"] = [1, 2]
buttons["i"] = [0, 5]
buttons["j"] = [1, 4]
buttons["k"] = [1, 5]
buttons["l"] = [1, 6]
frames = {}
labels = {}
for key, button in buttons.items():
    frame = tk.Frame(
        master=window,
        relief=tk.RAISED,
        borderwidth=1
    )
    i, j = button
    frames[key] = frame
    frame.grid(row=i, column=j, padx=0, pady=0, sticky="nsew")
    frame.configure(bg="green")
    
    label = tk.Label(master=frame, text=key, font=("Arial", 16, "bold"))
    label.pack(padx=2, pady=2)
    label.configure(bg="Green")
    labels[key] = label
def key_change_color(event):
    if event.keysym in frames:
        frames[event.keysym].configure(bg="Red")
        labels[event.keysym].configure(bg="Red")
    else:
        print("Unknown key:", event.keysym)
def key_restore_color(event):
    if event.keysym in frames:
        frames[event.keysym].configure(bg="Green")
        labels[event.keysym].configure(bg="Green")
    else:
        print("Unknown key:", event.keysym)
window.bind("<KeyPress>", key_change_color)
window.bind("<KeyRelease>", key_restore_color)
window.mainloop()