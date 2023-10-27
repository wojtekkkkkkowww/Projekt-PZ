import tkinter as tk
from menu import ChessMenu

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("400x400")
    root.title("Chess App")
    
    menu = ChessMenu(root)
    menu.grid(row=0, column=0, sticky="nsew")
    
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    
    root.mainloop()