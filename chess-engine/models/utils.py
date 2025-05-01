import torch
import chess
import random

def board_to_tensor(board):
    """Convert a chess board to a tensor representation."""
    pieces_to_index = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    tensor = torch.zeros(13, 8, 8)

    # Fill the 13th channel with 1's if it's white to move, 0's if it's black
    if board.turn == chess.WHITE:
        tensor[12].fill_(1.0)

    # Fill piece channels
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            color_idx = 0 if piece.color == chess.WHITE else 6
            piece_idx = color_idx + pieces_to_index[piece.piece_type]

            file_idx = chess.square_file(square)
            rank_idx = 7 - chess.square_rank(square)

            tensor[piece_idx][rank_idx][file_idx] = 1.0

    return tensor

def evaluate_position(board, model):
    """Evaluate a position using the neural network model."""
    model.eval()
    with torch.no_grad():
        tensor = board_to_tensor(board)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        evaluation = model(tensor).item()

        # Flip evaluation if it's black's turn (model is trained from white's perspective)
        if board.turn == chess.BLACK:
            evaluation = -evaluation

        return evaluation

def minimax(board, depth, alpha, beta, is_maximizing, model, position_history=None):
    """Minimax algorithm with alpha-beta pruning."""
    if depth == 0 or board.is_game_over():
        return evaluate_position(board, model)

    # Check for threefold repetition
    if position_history:
        current_fen = board.fen().split(' ')[0]  # Just the piece positions
        if sum(1 for fen in position_history if fen.split(' ')[0] == current_fen) >= 2:
            return 0  # Draw evaluation for repetition

    if is_maximizing:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            if position_history is not None:
                position_history.append(board.fen())

            eval = minimax(board, depth - 1, alpha, beta, False, model, position_history)

            if position_history is not None:
                position_history.pop()
            board.pop()

            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            if position_history is not None:
                position_history.append(board.fen())

            eval = minimax(board, depth - 1, alpha, beta, True, model, position_history)

            if position_history is not None:
                position_history.pop()
            board.pop()

            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(board, model, depth=3, position_history=None):
    """Find the best move using minimax with alpha-beta pruning."""
    best_move = None
    best_eval = float('-inf')
    alpha = float('-inf')
    beta = float('inf')

    # For early game, add some randomness to the moves
    if board.fullmove_number < 5:
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return random.choice(legal_moves)

    # Iterate through all legal moves
    for move in board.legal_moves:
        board.push(move)
        if position_history is not None:
            position_history.append(board.fen())

        # Evaluate this move
        eval = minimax(board, depth - 1, alpha, beta, False, model, position_history)

        if position_history is not None:
            position_history.pop()
        board.pop()

        # Update best move if found
        if eval > best_eval:
            best_eval = eval
            best_move = move

        # Update alpha
        alpha = max(alpha, eval)

    return best_move

def result_to_value(result, turn):
    if result == '*':  # Game unfinished
        return 0.0

    if result == '1-0':  # White win
        return 1.0 if turn else -1.0
    elif result == '0-1':  # Black win
        return -1.0 if turn else 1.0
    else:  # Draw ('1/2-1/2')
        return 0.0
