import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import chess
import chess.pgn
import numpy as np

# Simplified CNN model
class ChessModel(nn.Module):
    def __init__(self):
        super(ChessModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(12, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Simplified board to tensor conversion
def board_to_tensor(board):
    tensor = torch.zeros(12, 8, 8)
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # For white pieces: channels 0-5, for black: channels 6-11
            idx = piece_types.index(piece.piece_type)
            if piece.color == chess.BLACK:
                idx += 6

            # Convert square to coordinates
            rank = 7 - chess.square_rank(square)
            file = chess.square_file(square)
            tensor[idx][rank][file] = 1.0

    return tensor

# Simplified dataset class
class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, pgn_dir, max_games=5000, max_positions=50000, min_elo=0, sample_rate=0.1):
        self.positions = []
        self.results = []
        positions_added = 0
        games_processed = 0

        # Find all PGN files in the directory
        pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith('.pgn')]
        print(f"Found {len(pgn_files)} PGN files in {pgn_dir}")

        for pgn_file in pgn_files:
            file_path = os.path.join(pgn_dir, pgn_file)
            print(f"Loading positions from {file_path}...")

            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                while games_processed < max_games:
                    try:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break

                        # Check Elo ratings if min_elo is specified
                        if min_elo > 0:
                            white_elo = int(game.headers.get("WhiteElo", "0") or "0")
                            black_elo = int(game.headers.get("BlackElo", "0") or "0")
                            if white_elo < min_elo or black_elo < min_elo:
                                continue

                        # Get game result
                        if game.headers.get("Result", "*") == "1-0":
                            result = 1.0
                        elif game.headers.get("Result", "*") == "0-1":
                            result = -1.0
                        else:
                            result = 0.0

                        games_processed += 1
                        if games_processed % 100 == 0:
                            print(f"Processed {games_processed} games, extracted {positions_added} positions")

                        # Process game positions
                        board = game.board()
                        for move in game.mainline_moves():
                            board.push(move)

                            # Skip early moves and only sample some positions
                            if board.fullmove_number < 5 or np.random.random() > sample_rate:
                                continue

                            # Convert board to tensor and store with result
                            tensor = board_to_tensor(board)
                            self.positions.append(tensor)

                            # Adjust result based on who's turn it is
                            adjusted_result = result if board.turn == chess.WHITE else -result
                            self.results.append(torch.tensor([adjusted_result], dtype=torch.float32))
                            positions_added += 1

                            if positions_added >= max_positions:
                                break

                        if positions_added >= max_positions:
                            break

                    except Exception as e:
                        print(f"Error processing game: {e}")
                        continue

            if games_processed >= max_games or positions_added >= max_positions:
                break

        print(f"Dataset created with {len(self.positions)} positions from {games_processed} games")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.positions[idx], self.results[idx]

# Simple training function
def train_model(pgn_dir, output_path, max_games=5000, max_positions=50000,
               epochs=10, batch_size=64, learning_rate=0.001, min_elo=0, sample_rate=0.1):
    # Create dataset and split into train/validation
    dataset = ChessDataset(pgn_dir, max_games, max_positions, min_elo, sample_rate)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ChessModel().to(device)

    # Training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for positions, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            positions = positions.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(positions)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for positions, targets in val_loader:
                positions = positions.to(device)
                targets = targets.to(device)
                outputs = model(positions)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_path)
            print(f"Model saved with validation loss: {best_val_loss:.4f}")

    print("Training complete!")
    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a simple chess evaluation model")
    parser.add_argument("--pgn_dir", type=str, required=True, help="Directory containing PGN files")
    parser.add_argument("--output", type=str, default="trained_models/chess_model.pth", help="Output model path")
    parser.add_argument("--max_games", type=int, default=5000, help="Maximum number of games to process")
    parser.add_argument("--max_positions", type=int, default=50000, help="Maximum positions to extract")
    parser.add_argument("--min_elo", type=int, default=0, help="Minimum player ELO to include games")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--sample_rate", type=float, default=0.1, help="Rate to sample positions (0.0-1.0)")

    args = parser.parse_args()

    train_model(
        pgn_dir=args.pgn_dir,
        output_path=args.output,
        max_games=args.max_games,
        max_positions=args.max_positions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_elo=args.min_elo,
        sample_rate=args.sample_rate
    )
