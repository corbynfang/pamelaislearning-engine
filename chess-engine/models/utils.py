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

def minimax(board, depth, alpha, beta, maximizing_player, model, position_history=None):
    # Check for game over
    if board.is_game_over():
        if board.is_checkmate():
            return -10000 if maximizing_player else 10000
        # Draw
        return 0

    # Add position repetition detection
    if position_history is not None:
        current_pos = board.fen().split(' ')[0]  # Just the piece positions
        pos_count = 0
        for pos in position_history:
            if pos.split(' ')[0] == current_pos:
                pos_count += 1
                if pos_count >= 2:  # We're about to repeat a position a third time
                    return 0  # Avoid repetition

    # Reached the maximum depth, evaluate with quiescence search instead of just static eval
    if depth == 0:
        return quiescence_search(board, model, alpha, beta)

    if maximizing_player:
        max_eval = float('-inf')

        # Move ordering - examine captures first
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: 10 if board.is_capture(m) else 0, reverse=True)

        for move in moves:
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

        # Move ordering - examine captures first
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: 10 if board.is_capture(m) else 0, reverse=True)

        for move in moves:
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

def find_best_move(board, model, depth=5, position_history=None):

    best_move = None
    best_eval = float('-inf')
    alpha = float('-inf')
    beta = float('inf')

    # For early game, consider standard opening moves rather than pure randomness
    if board.fullmove_number < 5:
        # Standard opening book for first few moves
        if board.fen().split(' ')[0] == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR" and board.turn == chess.WHITE:
            # Starting position as White - use common openings
            common_first_moves = [
                chess.Move.from_uci("e2e4"),  # King's Pawn
                chess.Move.from_uci("d2d4"),  # Queen's Pawn
                chess.Move.from_uci("c2c4"),  # English Opening
                chess.Move.from_uci("g1f3")   # Reti Opening
            ]
            for move in common_first_moves:
                if move in board.legal_moves:
                    return move

        # Or if responding to 1.e4 as Black
        if board.fen().split(' ')[0] == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR" and board.turn == chess.BLACK:
            common_responses = [
                chess.Move.from_uci("e7e5"),  # Open Game
                chess.Move.from_uci("c7c5"),  # Sicilian Defense
                chess.Move.from_uci("e7e6")   # French Defense
            ]
            for move in common_responses:
                if move in board.legal_moves:
                    return move

    # Get list of legal moves
    moves = list(board.legal_moves)

    # Move ordering - examine captures and checks first for better pruning
    ordered_moves = []
    for move in moves:
        # Give captures a higher priority (simple MVV-LVA)
        if board.is_capture(move):
            ordered_moves.append((move, 10))
        # Then look at checks
        else:
            # Check if move gives check
            board.push(move)
            gives_check = board.is_check()
            board.pop()

            if gives_check:
                ordered_moves.append((move, 5))
            else:
                ordered_moves.append((move, 0))

    # Sort moves by score, highest first
    ordered_moves.sort(key=lambda x: x[1], reverse=True)

    # Iterate through all legal moves in priority order
    for move, _ in ordered_moves:
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

    # If we somehow didn't find a best move, pick a random one
    if best_move is None and moves:
        best_move = random.choice(moves)

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

def quiescence_search(board, model, alpha, beta, depth=3):
    stand_pat = evaluate_position(board, model)

    if depth == 0:
        return stand_pat

    if stand_pat >= beta:
                return beta
    if alpha < stand_pat:
        alpha = stand_pat

            # Consider only capture moves
    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiescence_search(board, model, -beta, -alpha, depth-1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

    return alpha
