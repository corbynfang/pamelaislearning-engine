import chess
import torch
import random
import numpy as np
import time
from collections import deque
import os
from tqdm import tqdm
import chess.engine
from src.utils import board_to_tensor, evaluate_position, find_best_move
from models import ChessModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReinforcementLearning:
    def __init__(self, model_path=None, stockfish_path="stockfish", memory_size=10000,
                 batch_size=64, gamma=0.95, learning_rate=0.001):
        # Initialize neural network model
        self.model = ChessModel(num_classes=1)
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
        self.model.to(DEVICE)

        # Initialize Stockfish engine
        self.stockfish_path = stockfish_path

        # Replay memory
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma

        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

        # Exploration parameters
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate

    def get_stockfish_eval(self, board, time_limit=0.1):
        """Get evaluation from Stockfish engine"""
        with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
            result = engine.analyse(board, chess.engine.Limit(time=time_limit))
            score = result["score"].white().score(mate_score=10000)
            # Normalize score to be between -1 and 1
            normalized_score = np.tanh(score / 1000) if score is not None else 0
            return normalized_score

    def choose_move(self, board, training=True):
        """Choose a move using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Exploration: choose a random move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves)
            return None
        else:
            # Exploitation: choose best move according to model
            return find_best_move(board, self.model, depth=2)

    def remember(self, state, action, reward, next_state, done):
        """Add experience to replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train model on batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Get prediction for next state
                next_state_tensor = board_to_tensor(next_state).unsqueeze(0).to(DEVICE)
                target += self.gamma * self.model(next_state_tensor).item()

            # Get current prediction
            state_tensor = board_to_tensor(state).unsqueeze(0).to(DEVICE)
            current_prediction = self.model(state_tensor)

            # Create target tensor
            target_tensor = torch.tensor([[target]], dtype=torch.float32).to(DEVICE)

            # Compute loss and update model
            self.optimizer.zero_grad()
            loss = self.criterion(current_prediction, target_tensor)
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def play_game_against_stockfish(self, elo_level=1500, max_moves=100, train=True):
        """Play a game against Stockfish and learn from it"""
        board = chess.Board()
        game_memory = []
        move_count = 0

        # Setup Stockfish with appropriate ELO
        stockfish_config = {"Skill Level": min(20, elo_level // 100)}  # Approximate mapping

        while not board.is_game_over() and move_count < max_moves:
            # Our model plays as white
            if board.turn == chess.WHITE:
                current_state = board.copy()
                move = self.choose_move(board, training=train)

                if move is None:  # No legal moves
                    break

                board.push(move)
                next_state = board.copy()

                # Calculate reward based on Stockfish evaluation change
                reward = -self.get_stockfish_eval(board)  # Negative because it's from opponent's perspective
                done = board.is_game_over()

                if train:
                    # Store experience for later training
                    game_memory.append((current_state, move, reward, next_state, done))

            # Stockfish plays as black
            else:
                with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                    engine.configure(stockfish_config)
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    board.push(result.move)

            move_count += 1

        # Calculate final reward based on game outcome
        if board.is_checkmate():
            final_reward = 1.0 if board.turn == chess.BLACK else -1.0  # We win if black is checkmated
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition():
            final_reward = 0.0  # Draw
        else:
            final_reward = 0.0  # Game didn't finish properly

        # Add experiences to memory with final reward
        if train and game_memory:
            for i, (state, move, reward, next_state, done) in enumerate(game_memory):
                # Discount reward based on how far from the end
                if i == len(game_memory) - 1:
                    adjusted_reward = final_reward
                else:
                    adjusted_reward = reward

                self.remember(state, move, adjusted_reward, next_state, done)

            # Train on replay memory
            self.replay()

        return board, final_reward

    def train(self, num_games=100, save_dir='models', elo_start=1200, elo_increment=50):
        """Train the model by playing multiple games against Stockfish with increasing difficulty"""
        results = {'wins': 0, 'losses': 0, 'draws': 0}
        elo = elo_start

        os.makedirs(save_dir, exist_ok=True)

        for game_num in tqdm(range(num_games), desc="Training"):
            board, reward = self.play_game_against_stockfish(elo_level=elo, train=True)

            # Record result
            if reward > 0:
                results['wins'] += 1
                # Increase difficulty after wins
                if results['wins'] % 5 == 0:
                    elo += elo_increment
                    print(f"Increasing difficulty to ELO {elo}")
            elif reward < 0:
                results['losses'] += 1
            else:
                results['draws'] += 1

            # Save model periodically
            if (game_num + 1) % 10 == 0:
                model_path = os.path.join(save_dir, f'reinforcement_model_game_{game_num+1}.pth')
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epsilon': self.epsilon,
                    'game_num': game_num,
                }, model_path)
                print(f"Model saved to {model_path}")
                print(f"Results after {game_num+1} games: {results}")

        # Save final model
        final_model_path = os.path.join(save_dir, 'reinforcement_model_final.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'game_num': num_games,
        }, final_model_path)

        print(f"Training completed. Final results: {results}")
        print(f"Final model saved to {final_model_path}")
        return results
