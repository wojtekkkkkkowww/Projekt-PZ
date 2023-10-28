import tkinter as tk
import chess
import chess.svg
from chess_board import ChessBoard

class ChessGame(tk.Frame):
    
    def __init__(self, parent, with_bot=True):
        super().__init__(parent)
        self.parent = parent
        self.with_bot = with_bot
        self.current_color = chess.WHITE
        self.canvas = ChessBoard(self,self)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.start_game()

    def start_game(self):
        self.board = chess.Board()
        self.canvas.draw_board(self.board)
    
    def make_move(self,to_square):
        if self.canvas.highlighted_square is not None:
            move = chess.Move(self.canvas.highlighted_square, to_square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.canvas.highlighted_square = None
                self.current_color = not self.current_color
                self.canvas.draw_board(self.board)
            else:
                self.canvas.unhighlight_square()
                self.canvas.highlighted_square = None
                