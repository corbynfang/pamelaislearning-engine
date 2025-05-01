import chess
import chess.engine
import pygame
import time
import sys
import os
from models.models import ChessModel
from models.utils import board_to_tensor, find_best_move
import torch
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

models_dir = os.path.join(current_dir, 'models')
init_file = os.path.join(models_dir, '__init__.py')
if not os.path.exists(init_file):
    os.makedirs(models_dir, exist_ok=True)
    with open(init_file, 'w') as f:
        pass

print(f"Python path: {sys.path}")
print(f"Current directory: {current_dir}")
print(f"Models directory exists: {os.path.exists(models_dir)}")

# Initialize Pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_SQUARE = (118, 150, 86)
LIGHT_SQUARE = (238, 238, 210)
HIGHLIGHT = (255, 255, 0, 50)  # Yellow with alpha
TEXT_COLOR = (50, 50, 50)

# Screen dimensions
SQUARE_SIZE = 60
BOARD_SIZE = 8 * SQUARE_SIZE
INFO_PANEL_WIDTH = 300
WINDOW_WIDTH = BOARD_SIZE + INFO_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE
FONT_SIZE = 16

class ChessGUI:
    def __init__(self, model_path, stockfish_path, thinking_time=0.5, elo=1500):
        # Setup the window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess AI vs Stockfish")

        # Load fonts
        self.font = pygame.font.SysFont('Arial', FONT_SIZE)
        self.big_font = pygame.font.SysFont('Arial', 24)

        # Load chess pieces images
        self.pieces_images = {}
        for piece in ['p', 'n', 'b', 'r', 'q', 'k']:
            # Load white pieces
            if os.path.exists(f"pieces/{piece.upper()}.png"):
                img = pygame.image.load(f"pieces/{piece.upper()}.png")
            else:
                img = self._create_text_image(piece.upper())
            self.pieces_images[piece.upper()] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))

            # Load black pieces
            if os.path.exists(f"pieces/{piece}.png"):
                img = pygame.image.load(f"pieces/{piece}.png")
            else:
                img = self._create_text_image(piece)
            self.pieces_images[piece] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))

        # Setup the board
        self.board = chess.Board()

        # Load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessModel(num_classes=1)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Setup Stockfish
        self.stockfish_path = stockfish_path
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"Skill Level": min(20, elo // 100)})

        # Game variables
        self.thinking_time = thinking_time
        self.elo = elo
        self.last_move = None
        self.game_over = False
        self.ai_evaluation = 0.0
        self.stockfish_evaluation = 0.0
        self.move_history = []

    def _create_text_image(self, piece_symbol):
        """Create a simple text image if piece images are missing"""
        surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(surface, (200, 200, 200, 150), (0, 0, SQUARE_SIZE, SQUARE_SIZE))
        font = pygame.font.SysFont('Arial', 48)
        text = font.render(piece_symbol, True, BLACK)
        surface.blit(text, (SQUARE_SIZE//2 - text.get_width()//2, SQUARE_SIZE//2 - text.get_height()//2))
        return surface

    def draw_board(self):
        """Draw the chess board and pieces"""
        # Clear the screen
        self.screen.fill(WHITE)

        # Draw the squares
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

        # Draw coordinates
        for i in range(8):
            # Draw rank numbers (8-1)
            rank_text = self.font.render(str(8-i), True, BLACK if i % 2 == 0 else WHITE)
            self.screen.blit(rank_text, (5, i * SQUARE_SIZE + 5))

            # Draw file letters (a-h)
            file_text = self.font.render(chr(97 + i), True, BLACK if (i + 7) % 2 == 0 else WHITE)
            self.screen.blit(file_text, (i * SQUARE_SIZE + SQUARE_SIZE - 15, BOARD_SIZE - 20))

        # Highlight last move
        if self.last_move:
            from_square = self.last_move.from_square
            to_square = self.last_move.to_square

            from_col, from_row = chess.square_file(from_square), 7 - chess.square_rank(from_square)
            to_col, to_row = chess.square_file(to_square), 7 - chess.square_rank(to_square)

            # Create a transparent surface for highlighting
            highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            highlight_surface.fill(HIGHLIGHT)

            # Blit the highlight surfaces
            self.screen.blit(highlight_surface, (from_col * SQUARE_SIZE, from_row * SQUARE_SIZE))
            self.screen.blit(highlight_surface, (to_col * SQUARE_SIZE, to_row * SQUARE_SIZE))

        # Draw pieces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                col = chess.square_file(square)
                row = 7 - chess.square_rank(square)  # Flip because chess board has 1st rank at bottom
                piece_img = self.pieces_images.get(piece.symbol())
                if piece_img:
                    self.screen.blit(piece_img, (col * SQUARE_SIZE, row * SQUARE_SIZE))

        # Draw info panel
        self.draw_info_panel()

        # Update the display
        pygame.display.flip()

    def draw_info_panel(self):
        """Draw the information panel"""
        # Panel background
        pygame.draw.rect(self.screen, (240, 240, 240), (BOARD_SIZE, 0, INFO_PANEL_WIDTH, WINDOW_HEIGHT))
        pygame.draw.line(self.screen, BLACK, (BOARD_SIZE, 0), (BOARD_SIZE, WINDOW_HEIGHT), 2)

        # Game info
        title = self.big_font.render("Chess AI vs Stockfish", True, BLACK)
        self.screen.blit(title, (BOARD_SIZE + 20, 20))

        # Current turn
        turn_text = f"Turn: {'White (Your AI)' if self.board.turn else 'Black (Stockfish)'}"
        turn_surface = self.font.render(turn_text, True, TEXT_COLOR)
        self.screen.blit(turn_surface, (BOARD_SIZE + 20, 60))

        # Move number
        move_text = f"Move #: {self.board.fullmove_number}"
        move_surface = self.font.render(move_text, True, TEXT_COLOR)
        self.screen.blit(move_surface, (BOARD_SIZE + 20, 85))

        # Last move
        last_move_text = f"Last move: {self.last_move}" if self.last_move else "No moves yet"
        last_move_surface = self.font.render(last_move_text, True, TEXT_COLOR)
        self.screen.blit(last_move_surface, (BOARD_SIZE + 20, 110))

        # Evaluation
        eval_text = f"AI evaluation: {self.ai_evaluation:.2f}"
        eval_surface = self.font.render(eval_text, True, TEXT_COLOR)
        self.screen.blit(eval_surface, (BOARD_SIZE + 20, 145))

        # Game status
        status_y = 180
        if self.board.is_check():
            check_text = "CHECK!"
            check_surface = self.font.render(check_text, True, (255, 0, 0))
            self.screen.blit(check_surface, (BOARD_SIZE + 20, status_y))
            status_y += 25

        if self.game_over:
            result_text = f"Game over! Result: {self.board.result()}"
            result_surface = self.big_font.render(result_text, True, (0, 0, 200))
            self.screen.blit(result_surface, (BOARD_SIZE + 20, status_y))
            status_y += 30

            if self.board.is_checkmate():
                end_text = "CHECKMATE!"
            elif self.board.is_stalemate():
                end_text = "STALEMATE!"
            elif self.board.is_insufficient_material():
                end_text = "INSUFFICIENT MATERIAL!"
            else:
                end_text = "GAME ENDED"

            end_surface = self.font.render(end_text, True, (0, 0, 200))
            self.screen.blit(end_surface, (BOARD_SIZE + 20, status_y))

        # Move history
        history_y = 270
        history_title = self.font.render("Move History:", True, BLACK)
        self.screen.blit(history_title, (BOARD_SIZE + 20, history_y))
        history_y += 25

        # Display last 10 moves
        for idx, move in enumerate(self.move_history[-10:]):
            move_text = f"{idx + len(self.move_history) - 9}. {move}"
            move_surface = self.font.render(move_text, True, TEXT_COLOR)
            self.screen.blit(move_surface, (BOARD_SIZE + 20, history_y))
            history_y += 20

    def get_ai_evaluation(self):
        """Get evaluation from your AI"""
        with torch.no_grad():
            tensor = board_to_tensor(self.board).unsqueeze(0).to(self.device)
            evaluation = self.model(tensor).item()
            if self.board.turn == chess.BLACK:
                evaluation = -evaluation
            return evaluation

    def play_move(self):
        """Play a single move in the game"""
        if self.board.is_game_over():
            self.game_over = True
            return

        # Get AI evaluation
        self.ai_evaluation = self.get_ai_evaluation()

        # AI's turn (White)
        if self.board.turn == chess.WHITE:
            pygame.display.set_caption("Chess AI vs Stockfish - AI Thinking...")
            self.draw_board()

            # AI makes a move
            move = find_best_move(self.board, self.model, depth=3)
            self.last_move = move
            self.board.push(move)
            self.move_history.append(str(move))

            pygame.display.set_caption("Chess AI vs Stockfish")

        # Stockfish's turn (Black)
        else:
            pygame.display.set_caption("Chess AI vs Stockfish - Stockfish Thinking...")
            self.draw_board()

            # Stockfish makes a move
            result = self.engine.play(self.board, chess.engine.Limit(time=self.thinking_time))
            self.last_move = result.move
            self.board.push(result.move)
            self.move_history.append(str(result.move))

            pygame.display.set_caption("Chess AI vs Stockfish")

        # Check if the game is over
        if self.board.is_game_over():
            self.game_over = True

    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()

        try:
            running = True
            paused = False

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        elif event.key == pygame.K_r:
                            # Reset the game
                            self.board = chess.Board()
                            self.last_move = None
                            self.game_over = False
                            self.move_history = []

                self.draw_board()

                if not paused and not self.game_over:
                    self.play_move()
                    time.sleep(0.5)  # Add a small delay to make it easier to follow

                clock.tick(30)  # 30 FPS

        finally:
            # Clean up
            pygame.quit()
            self.engine.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch your AI play against Stockfish in a GUI")
    parser.add_argument("--model", type=str, required=True, help="Path to your trained model")
    parser.add_argument("--stockfish", type=str, default="stockfish", help="Path to Stockfish executable")
    parser.add_argument("--time", type=float, default=0.5, help="Thinking time for Stockfish (seconds)")
    parser.add_argument("--elo", type=int, default=1500, help="Approximate ELO for Stockfish")

    args = parser.parse_args()

    # Create chess pieces folder if it doesn't exist (for custom piece images)
    if not os.path.exists("pieces"):
        os.makedirs("pieces")

    # Start the GUI
    chess_gui = ChessGUI(
        model_path=args.model,
        stockfish_path=args.stockfish,
        thinking_time=args.time,
        elo=args.elo
    )
    chess_gui.run()
