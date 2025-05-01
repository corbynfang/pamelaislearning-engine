import os
import sys
import torch.nn as nn
import chess
import torch
import numpy as np

# Add path to your source directory
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
sys.path.append(src_dir)
sys.path.append(backend_dir)

# Import your model and functions
from models import ChessModel
from app import board_to_tensor, evaluate_position

def test_model(model_path):
    """Test if a model produces meaningful evaluations"""
    print(f"Loading model from: {model_path}")

    # Load the model
    model = ChessModel(num_classes=1)
    checkpoint = torch.load(model_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Test positions
    test_positions = [
        {
            "name": "Starting position",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        },
        {
            "name": "White has advantage",
            "fen": "rnbqkbnr/ppp2ppp/8/3pp3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3"
        },
        {
            "name": "Black has advantage",
            "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5Q2/PPPP1PPP/RNB1KBNR w KQkq - 0 3"
        },
        {
            "name": "White is winning",
            "fen": "4k3/8/8/8/8/8/PPPP4/RNBQKBNR w KQ - 0 1"
        },
        {
            "name": "Black is winning",
            "fen": "rnbqkbnr/pppp4/8/8/8/8/8/4K3 w - - 0 1"
        }
    ]

    print("\nDirect Model Output Tests:")
    print("=" * 50)

    # Raw output of model for each position
    for position in test_positions:
        board = chess.Board(position["fen"])
        tensor = board_to_tensor(board).to('cpu')

        with torch.no_grad():
            output = model(tensor)

        print(f"Position: {position['name']}")
        print(f"FEN: {position['fen']}")
        print(f"Raw model output: {output.item()}")
        print(f"Model output type: {type(output.item())}")
        print("-" * 50)

    # Check model parameters
    print("\nModel Parameter Information:")
    print("=" * 50)

    for name, param in model.named_parameters():
        print(f"Parameter: {name}")
        print(f"Shape: {param.shape}")
        print(f"Data range: {param.min().item()} to {param.max().item()}")
        print(f"Mean: {param.mean().item()}")
        print(f"Standard deviation: {param.std().item()}")
        print("-" * 30)

    # Check the final layer specifically
    print("\nFinal Layer Analysis:")
    print("=" * 50)

    final_layer = None
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.out_features == 1:
            final_layer = module
            break

    if final_layer:
        print("Final layer weights:")
        print(final_layer.weight.data)
        print("Final layer bias:")
        print(final_layer.bias.data)
    else:
        print("Could not identify final layer with 1 output!")

if __name__ == "__main__":
    # Get model path from user
    model_name = input("Enter the model filename (e.g., final_chess_model.pth): ")
    model_path = os.path.join(src_dir, "models", model_name)

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        model_path = input("Enter full path to model: ")

    test_model(model_path)
