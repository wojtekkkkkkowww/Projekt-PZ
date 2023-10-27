import tkinter as tk
from chess_game import ChessGame


class ChessMenu(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        play_with_bot_button = tk.Button(self, text="Play with Bot", command=self.show_game_board_with_bot)
        two_players_mode_button = tk.Button(self, text="Two Players Mode", command=self.show_game_board_two_players)
        exit_button = tk.Button(self, text="Exit", command=self.parent.quit)

        play_with_bot_button.place(relx=0.5, rely=0.3, anchor="center")
        two_players_mode_button.place(relx=0.5, rely=0.5, anchor="center")
        exit_button.place(relx=0.5, rely=0.7, anchor="center")

    def show_game_board_with_bot(self):
        self.pack_forget()  
        ChessGame(self.parent, with_bot=True).grid(row=1, column=0, sticky="nsew")  

    def show_game_board_two_players(self):
        self.pack_forget()  
        ChessGame(self.parent, with_bot=False).grid(row=1, column=0, sticky="nsew")  

