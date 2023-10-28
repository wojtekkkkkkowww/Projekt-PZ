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
        self.selected_square = None
        self.start_game()

    def start_game(self):
        self.board = chess.Board()
        self.canvas.draw_board(self.board)

    def select_square(self, square):
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece is not None and piece.color == self.current_color:
                self.selected_square = square
                self.canvas.highlight_square(square, "green")

    
    def make_move(self,to_square):
        if self.selected_square is not None:
            move = chess.Move(self.selected_square, to_square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.current_color = not self.current_color
                self.canvas.draw_board(self.board)