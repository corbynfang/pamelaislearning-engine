from models.utils import evaluate_position
from mypyc.test-data.fixtures.ir import eval
import os
import chess
import torch
import berserk
import chess.polyglot
import chess.gaviota
from dotenv import load_dotenv
from train import ChessModel, board_to_tensor


class LichessBot:
    def __init__(self):
        load_dotenv()
        # Setup Lichess client
        self.token = os.getenv('LICHESS_API_TOKEN')
        self.session = berserk.TokenSession(self.token)
        self.client = berserk.Client(self.session)

        # Load the trained model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessModel().to(self.device)
        model_path = "../trained_models/final_chess_model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("Model loaded successfully!")

        try:
            self.opening_book = chess.polyglot.open_reader("../books/human.bin")
            print("Opening book loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load opening book: {e}")
            self.opening_book = None

    def get_book_move(self, board):
        """Try to get a move from the opening book"""
        if self.opening_book:
            try:
                entries = list(self.opening_book.find_all(board))
                if entries:
                    # Choose the move with the highest weight
                    entry = max(entries, key=lambda e: e.weight)
                    print(f"Book move found: {entry.move.uci()} (weight: {entry.weight})")
                    return entry.move
            except Exception as e:
                print(f"Error getting book move: {e}")
        return None

    def get_best_move(self, board):
        # First try to get a move from the opening book
        book_move = self.get_book_move(board)
        if book_move:
            return book_move

        # If no book move found, use your trained model
        legal_moves = list(board.legal_moves)
        best_move = None
        best_eval = float('-inf') if board.turn else float('inf')

        for move in legal_moves:
            board.push(move)
            tensor = board_to_tensor(board).unsqueeze(0).to(self.device)

            with torch.no_grad():
                eval = self.model(tensor).item()

            board.pop()

            if board.turn:  # White's turn
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
            else:  # Black's turn
                if eval < best_eval:
                    best_eval = eval
                    best_move = move

        return best_move

    def handle_game_stream(self):
        print("Waiting for games...")
        for event in self.client.bots.stream_incoming_events():
            if event['type'] == 'challenge':
                print(f"Received challenge from {event['challenge']['challenger']['name']}")
                self.handle_challenge(event['challenge'])
            elif event['type'] == 'gameStart':
                print(f"Game starting: {event['game']['id']}")
                self.handle_game(event['game']['id'])

    def handle_challenge(self, challenge):
        try:
            self.client.bots.accept_challenge(challenge['id'])
            print(f"Accepted challenge: {challenge['id']}")
        except berserk.exceptions.ResponseError as e:
            print(f"Could not accept challenge: {e}")

    def handle_game(self, game_id):

        game_info = self.client.games.get_ongoing()[0]
        clock = game_info.get('clock', {})
        initial_time = clock.get('initial', 60) * 1000  # Convert to milliseconds
        increment = clock.get('increment', 0) * 1000
        def get_think_time(remaining_time):
                # Use about 1/30th of remaining time, plus half the increment
            return (remaining_time / 30) + (increment / 2)

        print(f"Starting game: {game_id}")

        # Get game info to determine our color
        game_info = self.client.games.get_ongoing()[0]  # Get the current game
        is_white = game_info['color'] == 'white'
        print(f"Playing as {'white' if is_white else 'black'}")

        board = chess.Board()

        # If we're white, make the first move immediately
        if is_white and not board.move_stack:
            move = self.get_best_move(board)
            if move:
                self.client.bots.make_move(game_id, move.uci())
                print(f"Made first move as white: {move.uci()}")
                board.push(move)

        for event in self.client.bots.stream_game_state(game_id):
            if event['type'] == 'gameState':
                # Update board with all moves
                moves = event['moves'].split() if event['moves'] else []

                # Reset board and apply all moves
                board = chess.Board()
                for move in moves:
                    board.push(chess.Move.from_uci(move))

                # Check if it's our turn
                is_our_turn = (board.turn == chess.WHITE) == is_white

                if not board.is_game_over() and is_our_turn:
                    print(f"Making move for game {game_id}")
                    move = self.get_best_move(board)
                    if move:
                        try:
                            self.client.bots.make_move(game_id, move.uci())
                            print(f"Made move: {move.uci()}")
                            board.push(move)
                        except Exception as e:
                            print(f"Error making move: {e}")

            elif event['type'] == 'gameFinish':
                print(f"Game finished: {game_id}")
                break

    def evaluate_position(self, board):
        if board.is_checkmate():
            return -10000 if board.turn else 10000
        if board.is_stalemate():
            return 0

        score = 0

        score += self.count_material(board)
        score += self.evaluate_mobility(board)
        score += self.evaluate_pawn_structure(board)
        score += self.evaluate_king_safety(board)
        score += self.evaluate_center_control(board)

        return score if board.turn else -score

def main():
    print("Starting bot...")
    bot = LichessBot()
    while True:
        try:
            bot.handle_game_stream()
        except Exception as e:
            print(f"Error in game stream: {e}")
            continue

if __name__ == "__main__":
    main()
