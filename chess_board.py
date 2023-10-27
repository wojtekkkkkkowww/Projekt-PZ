import tkinter as tk
import chess
import chess.svg
import cairosvg
from PIL import Image, ImageTk
import io

darker_brown = "#8B4513"
light_cream = "#FFE4B5"
light_green = "#90EE90"

class ChessBoard(tk.Canvas):
    def __init__(self, parent,game):
        super().__init__(parent, width=400, height=400)
        self.piece_images = self.load_piece_images()
        self.game = game 
        self.bind("<Button-1>", self.on_square_click)
        self.highlighted_square = None 

    def load_piece_images(self):
        piece_images = {}
        for piece_symbol in "RNBQKPrnbqkp":
            svg_data = chess.svg.piece(chess.Piece.from_symbol(piece_symbol), size=40)
            png_data = cairosvg.svg2png(bytestring=svg_data.encode("UTF-8"))
            image = Image.open(io.BytesIO(png_data))
            tk_image = ImageTk.PhotoImage(image)
            piece_images[piece_symbol] = tk_image
        return piece_images


    def highlight_square(self, square, color):
        piece =  self.game.board.piece_at(square)
        rank, file = chess.square_rank(square), chess.square_file(square)
        x0, y0, x1, y1 = file * 50, rank * 50, (file + 1) * 50, (rank + 1) * 50
        self.create_rectangle(x0, y0, x1, y1, fill=color)
        if piece:
            x, y = file * 50 , (rank) * 50 
            piece_symbol = piece.symbol()
            self.create_image(x + 25, y + 25, image=self.piece_images[piece_symbol])


    def draw_board(self,board):
        

        for square in chess.SQUARES:
            rank, file = chess.square_rank(square), chess.square_file(square)
            color = darker_brown if (rank + file) % 2 == 0 else light_cream
            x0, y0, x1, y1 = file * 50, rank * 50, (file + 1) * 50, (rank + 1) * 50
            self.create_rectangle(x0, y0, x1, y1, fill=color)

        for square, piece in board.piece_map().items():
            piece_symbol = piece.symbol()
            x, y = chess.square_file(square) * 50, (chess.square_rank(square)) * 50
            self.create_image(x + 25, y + 25, image=self.piece_images[piece_symbol])

    def on_square_click(self, event):
        col, row = event.x // 50, event.y // 50
        square = chess.square(col,row)  

        if self.game is not None:
            if self.highlighted_square is not None:
                rank, file = chess.square_rank(self.highlighted_square), chess.square_file(self.highlighted_square)
                color = darker_brown if (rank + file) % 2 == 0 else light_cream
                self.highlight_square(self.highlighted_square,color) 
            self.game.select_square(square)
            self.highlight_square(square, light_green)
            self.highlighted_square = square  
