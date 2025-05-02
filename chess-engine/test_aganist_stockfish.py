import os
import sys
import chess
import chess.engine
import torch
import time
import random

# Add the paths to your modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import your modules
from models.models import ChessModel
from models.utils import board_to_tensor, evaluate_position, minimax, find_best_move

def test_single_game(model, stockfish_path, depth=3, our_color=chess.WHITE):
    """Play a single game against Stockfish and return the result"""
    # Set up Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": 2})  # Adjust Stockfish strength (1-20)

    # Set up the board
    board = chess.Board()

    position_history = []

    # For logging the game
    moves = []

    try:
        # Play until game is over
        while not board.is_game_over():
            print(f"Current position: {board.fen()}")
            position_history.append(board.fen())

            if board.turn == our_color:
                # Our engine's turn
                start_time = time.time()
                best_move = find_best_move(board, model, depth, position_history)
                end_time = time.time()

                if best_move:
                    print(f"Our engine plays: {best_move.uci()} (time: {end_time-start_time:.2f}s)")

                    # Verify move is legal
                    if best_move in board.legal_moves:
                        board.push(best_move)
                        moves.append(best_move.uci())
                    else:
                        print(f"ERROR: Engine returned illegal move {best_move.uci()}!")
                        print(f"Legal moves: {[m.uci() for m in board.legal_moves]}")
                        # Choose a random legal move instead
                        if board.legal_moves:
                            random_move = list(board.legal_moves)[0]
                            print(f"Playing random move {random_move.uci()} instead")
                            board.push(random_move)
                            moves.append(random_move.uci())
                        else:
                            print("No legal moves available - game should be over!")
                            break
                else:
                    print("Our engine couldn't find a move!")
                    break
            else:
                # Stockfish's turn
                print("Stockfish thinking...")
                result = engine.play(board, chess.engine.Limit(time=0.1))
                print(f"Stockfish plays: {result.move.uci()}")
                board.push(result.move)
                moves.append(result.move.uci())

            # Print evaluation after each move
            try:
                evaluation = evaluate_position(board, model)
                print(f"Position evaluation: {evaluation:.2f}")
            except Exception as e:
                print(f"Error evaluating position: {e}")

            print("-" * 40)

            # Check for threefold repetition to avoid loops
            if len(position_history) > 8:  # Only check after some moves
                position_counts = {}
                for pos in position_history:
                    board_pos = pos.split(' ')[0]
                    position_counts[board_pos] = position_counts.get(board_pos, 0) + 1
                    if position_counts[board_pos] >= 3:
                        print("Threefold repetition detected! Ending game.")
                        return "draw"

    except KeyboardInterrupt:
        print("Game interrupted!")
    except Exception as e:
        print(f"Error during game: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        engine.quit()

    # Game result
    result = board.result()
    print(f"Game over: {result}")
    print(f"Moves: {' '.join(moves)}")

    # Determine if our engine won
    if (result == "1-0" and our_color == chess.WHITE) or (result == "0-1" and our_color == chess.BLACK):
        return "win"
    elif (result == "0-1" and our_color == chess.WHITE) or (result == "1-0" and our_color == chess.BLACK):
        return "loss"
    else:
        return "draw"

def run_stockfish_tournament(model_path, num_games=2, stockfish_path="stockfish"):
    """Run a tournament against Stockfish with multiple games"""
    # Load the model
    print(f"Loading model from: {model_path}")

    model = ChessModel(num_classes=1)
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Results tracking
    results = {"win": 0, "loss": 0, "draw": 0}

    # Play games
    for game_num in range(num_games):
        print(f"\n=== Game {game_num+1}/{num_games} ===")
        # Alternate colors
        our_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK
        color_name = "White" if our_color == chess.WHITE else "Black"
        print(f"Our engine plays as {color_name}")

        # Play the game
        result = test_single_game(model, stockfish_path, depth=3, our_color=our_color)
        results[result] += 1

        # Print current standings
        print(f"\nTournament standings after {game_num+1} games:")
        print(f"Wins: {results['win']}, Losses: {results['loss']}, Draws: {results['draw']}")

    # Final results
    print("\n=== Tournament Results ===")
    print(f"Games played: {num_games}")
    print(f"Wins: {results['win']} ({results['win']/num_games:.1%})")
    print(f"Draws: {results['draw']} ({results['draw']/num_games:.1%})")
    print(f"Losses: {results['loss']} ({results['loss']/num_games:.1%})")

    return results

if __name__ == "__main__":
    # Path to your best model
    model_path = input("Enter model path (e.g., models/final_chess_model.pth): ")

    # Path to Stockfish executable
    stockfish_path = input("Enter path to Stockfish (default: stockfish): ") or "stockfish"

    # Number of games to play
    num_games = int(input("Number of games to play (default: 2): ") or "2")

    # Run a tournament
    run_stockfish_tournament(model_path, num_games=num_games, stockfish_path=stockfish_path)
