import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import chess
import chess.pgn
from models import ChessModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def board_to_tensor(board):
    pieces_to_index = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    tensor = torch.zeros(13, 8, 8)
    if board.turn == chess.WHITE:
        tensor[12].fill_(1.0)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            color_idx = 0 if piece.color == chess.WHITE else 6
            piece_idx = color_idx + pieces_to_index[piece.piece_type]
            file_idx = chess.square_file(square)
            rank_idx = 7 - chess.square_rank(square)
            tensor[piece_idx][rank_idx][file_idx] = 1.0
    return tensor

def load_pgn_games(pgn_dir, max_files=28, max_games_per_file=None):
    games = []
    files = [f for f in os.listdir(pgn_dir) if f.endswith('.pgn')]
    files = files[:min(len(files), max_files)]
    for file_name in tqdm(files, desc="Loading PGN files"):
        file_path = os.path.join(pgn_dir, file_name)
        loaded = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    games.append(game)
                    loaded += 1
                    if max_games_per_file and loaded >= max_games_per_file:
                        break
            print(f"Loaded {loaded} games from {file_name}")
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
    print(f"Total games loaded: {len(games)}")
    return games

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, games, max_positions=None):
        self.positions = []
        self.results = []
        positions_added = 0
        for game in tqdm(games, desc="Processing games"):
            result_str = game.headers.get("Result", "*")
            if result_str == "1-0":
                result = 1.0  # White wins
            elif result_str == "0-1":
                result = -1.0  # Black wins
            else:
                result = 0.0  # Draw
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if board.fullmove_number < 4:
                    continue
                tensor = board_to_tensor(board)
                adjusted_result = result if board.turn == chess.WHITE else -result

                # KEY FIX: Scale to match model output range
                scaled_result = adjusted_result * 3.0  # Scale to [-3, 3]

                self.positions.append(tensor)
                self.results.append(torch.tensor([scaled_result], dtype=torch.float32))
                positions_added += 1
                if max_positions and positions_added >= max_positions:
                    break
            if max_positions and positions_added >= max_positions:
                break
        print(f"Dataset created with {len(self.positions)} positions")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.results[idx]

def train_chess_model(pgn_dir, output_dir='../format/trained_models', max_files=28, max_games=100,
                     max_positions=100000, batch_size=32, epochs=10, learning_rate=0.0001):
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    print(f"Loading Games from {pgn_dir}..")
    games = load_pgn_games(pgn_dir, max_files, max_games)
    if not games:
        print("No games loaded, cannot train model.")
        return None

    dataset = ChessDataset(games, max_positions)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model and training components
    model = ChessModel(num_classes=1)
    model.to(DEVICE)

    # KEY FIX: Use standard MSE loss
    criterion = nn.MSELoss()

    # KEY FIX: More stable optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5, eps=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    best_val_loss = float('inf')
    nan_count_limit = 50  # Limit for NaN batches before skipping epoch

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batch_count = 0
        nan_batches = 0

        for batch_index, (positions, results) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Skip epoch if too many NaN batches
            if nan_batches > nan_count_limit:
                print(f"Too many NaN batches ({nan_batches}), skipping rest of epoch")
                break

            positions = positions.to(DEVICE)
            results = results.to(DEVICE)

            optimizer.zero_grad()

            try:
                outputs = model(positions)

                # Print diagnostic info for first batch
                if batch_index == 0:
                    print(f"Outputs range: [{outputs.min().item():.2f}, {outputs.max().item():.2f}]")
                    print(f"Targets range: [{results.min().item():.2f}, {results.max().item():.2f}]")

                # Skip batch if NaN detected
                if torch.isnan(outputs).any():
                    print(f"WARNING: NaN in outputs at batch {batch_index}, skipping")
                    nan_batches += 1
                    continue

                loss = criterion(outputs, results)

                if torch.isnan(loss).any():
                    print(f"WARNING: NaN in loss at batch {batch_index}, skipping")
                    nan_batches += 1
                    continue

                loss.backward()

                # KEY FIX: Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                batch_count += 1

                if (batch_index + 1) % 20 == 0:
                    print(f"Batch {batch_index + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error in batch {batch_index}: {e}")
                nan_batches += 1
                continue

        # Calculate average training loss
        avg_train_loss = train_loss / max(1, batch_count)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for positions, results in val_loader:
                try:
                    positions = positions.to(DEVICE)
                    results = results.to(DEVICE)
                    outputs = model(positions)

                    # Skip batch if NaN detected
                    if torch.isnan(outputs).any():
                        continue

                    loss = criterion(outputs, results)

                    if torch.isnan(loss).any():
                        continue

                    val_loss += loss.item()
                    val_batch_count += 1
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue

        # Calculate average validation loss
        avg_val_loss = val_loss / max(1, val_batch_count)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(output_dir, 'best_chess_model.pth'))
            print(f"✓ Model saved with validation loss: {best_val_loss:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_train_loss,
            }, os.path.join(output_dir, f'chess_model_epoch_{epoch+1}.pth'))

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
    }, os.path.join(output_dir, 'final_chess_model.pth'))

    print("Training complete!")
    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a chess position evaluation model")
    parser.add_argument("--pgn_dir", type=str, default="../data/pgn", help="Directory containing PGN files")
    parser.add_argument("--output_dir", type=str, default="../format/trained_models", help="Directory to save models")
    parser.add_argument("--max_files", type=int, default=28, help="Maximum number of PGN files to process")
    parser.add_argument("--max_games", type=int, default=10, help="Maximum games to load per file")
    parser.add_argument("--max_positions", type=int, default=5000, help="Maximum positions to extract")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")

    args = parser.parse_args()

    train_chess_model(
        pgn_dir=args.pgn_dir,
        output_dir=args.output_dir,
        max_files=args.max_files,
        max_games=args.max_games,
        max_positions=args.max_positions,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )
