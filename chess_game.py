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
    
    def enforce_promotion(self,move, board, promotion_piece):
        if move.promotion is None:
            from_square = move.from_square
            to_square = move.to_square
            if board.piece_at(from_square) == chess.Piece(chess.PAWN, board.turn):
                promotion_rank = 7 if board.turn == chess.WHITE else 0
                if chess.square_rank(to_square) == promotion_rank:
                    move.promotion = promotion_piece
        return move


    def make_move(self,to_square):
        if self.canvas.highlighted_square is not None:
            move = chess.Move(self.canvas.highlighted_square, to_square)
            move = self.enforce_promotion(move, self.board, chess.QUEEN)

            if move in self.board.legal_moves:
                self.board.push(move)
                self.canvas.highlighted_square = None
                self.current_color = not self.current_color
                self.canvas.draw_board(self.board)
           
            else:
                self.canvas.unhighlight_square()
                self.canvas.highlighted_square = None
                