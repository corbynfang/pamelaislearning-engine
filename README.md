# Pamelaislearning Chess Engine Documenation


# Project description
This is project is a chess engine made completely with python. This Documenation provides instructions for setting up and running Pamelaislearning a chess engine. The engine uses neural network-based evaluation and includes a GUI for playing aganist Stockfish if you want.

## Installation Guide

### Windows Installation

#### 1. Install Python of course.
1. Download Python 3.9 or newer from the [official website](https://www.python.org/downloads/windows/)
2. During installation, check "Add Python to PATH"
3. Complete the installation

#### 2. Setting up the Enviorment
Create a Virutal Enviorment (venv)
```powershell
python -m venv chess_engine_env

# Activate the enviorment
chess_engine_env\Scripts\activate
```

#### 3. Installing Requried Packages
```powershell
# install PyTorch and other dependencies
pip install torch torchvision torchaudio
pip install chess pygame numpy tqdm
```

#### 4. Installing Chess Data (https://theweekinchess.com/twic)
- Install the latest one that is available
- By the time this is commited the latest one is 1590

#### 5. Installing Stockfish Engine (if you want to use the engine to play aganist Stockfish)
1. Download Stockfish for Windows from the [official website](https://stockfishchess.org/download/)
2. Extract the .zip file to a folder of your choice
3. Remember the path to the stockfish.exe file (you'll need it later)

#### 5. Project Sturcture Setup
```powershell
# Create Project folders (if any problems i would just fork the project for Pamelaislearning chess engine)
mkdir chess-engine
mkdir format/trained_models
```

#### 6. Chess Piece Sprite
https://commons.wikimedia.org/wiki/File:Chess_Pieces_Sprite.svg

## Training The Model

### Windows (you can manipulate the signuartes if wanted)
```powershell
python train.py --pgn_dir data --output_dir format\trained_models --epochs 10 --batch_size 32
```

Options:
- `--pgn_dir`: Directory containing PGN files
- `--output_dir`: Directory to save trained models
- `--max_files`: Maximum number of PGN files to process
- `--max_games`: Maximum games to load per file
- `--max_positions`: Maximum positions to extract
- `--min_elo`: Minimum player ELO to include games
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--sample_rate`: Rate to sample positions (0.0-1.0)

## Testing the Model (in case I messed up the neural-networking logic)
```powershell
python test_model.py
```

## Playing aganist Stockfish with GUI

### Windows
```powershell
python view_game_gui.py --model format\trained_models\final_chess_model.pth --stockfish C:\path\to\stockfish.exe --time 0.5 --elo 1500
```

Options:
- `--model`: Path to your trained model
- `--stockfish`: Path to Stockfish executable
- `--time`: Thinking time for Stockfish in seconds
- `--elo`: Approximate ELO for Stockfish
- `--games`: Number of games to play

## Keyboard Controls (GUI)
- `Space`: Pause/resume the game
- `N`: Skip to next game
- `Q`: Quit all games
