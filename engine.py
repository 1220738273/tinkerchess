"""Simple chess engine: move generator + alpha-beta search.

This engine is intentionally self-contained so it can be imported without
starting the GUI. It implements basic move generation that mirrors the rules
in `tinker.py` (no castling/en-passant), a material-based evaluation and a
negamax alpha-beta search. Not a world-class engine, but solid for testing.

API:
  choose_move(board, color, depth) -> (from_sq, to_sq) algebraic strings or None

The board format is the same as in `tinker.py`: board[row][col] with row 0
== rank 8, row 7 == rank 1, pieces as 'wP','bK', etc., or None.
"""
from typing import List, Optional, Tuple
from tinker import algebraic_to_coords

PieceValues = {
    'P': 100,
    'N': 320,
    'B': 330,
    'R': 500,
    'Q': 900,
    'K': 20000,
}


class Engine:
    """Simple configurable engine instance.

    Holds piece values so we can tune them during training.
    """
    def __init__(self, piece_values: Optional[dict] = None):
        self.piece_values = piece_values.copy() if piece_values else PieceValues.copy()

    def evaluate(self, board: List[List[Optional[str]]]) -> int:
        """Evaluate board from White's perspective using current piece values."""
        score = 0
        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if not p:
                    continue
                val = self.piece_values.get(p[1], 0)
                if p[0] == 'w':
                    score += val
                else:
                    score -= val
        return score

    def choose_move(self, board: List[List[Optional[str]]], color: str, depth: int = 3) -> Optional[Tuple[str, str]]:
        # simple wrapper around module functions but using self.evaluate via closure
        # We'll implement a negamax that calls back to self.evaluate
        best_move = None

        def negamax(node_board, node_color, ply, alpha, beta):
            if ply == 0:
                return self.evaluate(node_board)
            moves = generate_moves(node_board, node_color)
            if not moves:
                return self.evaluate(node_board)
            value = -9999999
            for (fr, fc), (tr, tc) in moves:
                moved_piece = node_board[fr][fc]
                captured = node_board[tr][tc]
                make_move(node_board, fr, fc, tr, tc)
                score = -negamax(node_board, 'b' if node_color == 'w' else 'w', ply - 1, -beta, -alpha)
                undo_move(node_board, fr, fc, tr, tc, captured, moved_piece)
                if score > value:
                    value = score
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            return value

        moves = generate_moves(board, color)
        if not moves:
            return None
        best_score = -9999999
        alpha = -9999999
        beta = 9999999
        for (fr, fc), (tr, tc) in moves:
            moved_piece = board[fr][fc]
            captured = board[tr][tc]
            make_move(board, fr, fc, tr, tc)
            score = -negamax(board, 'b' if color == 'w' else 'w', depth - 1, -beta, -alpha)
            undo_move(board, fr, fc, tr, tc, captured, moved_piece)
            if score > best_score:
                best_score = score
                best_move = (coords_to_algebraic(fr, fc), coords_to_algebraic(tr, tc))
            alpha = max(alpha, score)
        return best_move

    def self_play(self, depth: int = 2, max_moves: int = 200) -> Tuple[str, List[Tuple[str, str]]]:
        """Play a game between two copies of this engine and return winner ('w','b','draw') and move list."""
        # initialize starting position
        board = [[None for _ in range(8)] for _ in range(8)]
        board[0] = ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"]
        board[1] = ["bP"] * 8
        for r in range(2, 6):
            board[r] = [None] * 8
        board[6] = ["wP"] * 8
        board[7] = ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
        moves = []
        turn = 'w'
        for ply in range(max_moves):
            mv = self.choose_move(board, turn, depth)
            if not mv:
                # no moves
                break
            fr, to = mv
            fr_r, fr_c = algebraic_to_coords(fr)
            tr_r, tr_c = algebraic_to_coords(to)
            make_move(board, fr_r, fr_c, tr_r, tr_c)
            moves.append((fr, to))
            # naive checkmate detection using is_checkmate from this module isn't available here
            # We'll just switch turn
            turn = 'b' if turn == 'w' else 'w'
        # simple evaluation to pick winner
        final = self.evaluate(board)
        if final > 0:
            return 'w', moves
        if final < 0:
            return 'b', moves
        return 'draw', moves




def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def is_path_clear(board: List[List[Optional[str]]], fr: int, fc: int, tr: int, tc: int) -> bool:
    dr = tr - fr
    dc = tc - fc
    step_r = _sign(dr)
    step_c = _sign(dc)
    if step_r == 0 and step_c == 0:
        return True
    r = fr + step_r
    c = fc + step_c
    while (r, c) != (tr, tc):
        if board[r][c] is not None:
            return False
        r += step_r
        c += step_c
    return True


def is_legal_move(board: List[List[Optional[str]]], fr: int, fc: int, tr: int, tc: int) -> bool:
    # bounds
    if not (0 <= fr < 8 and 0 <= fc < 8 and 0 <= tr < 8 and 0 <= tc < 8):
        return False
    piece = board[fr][fc]
    if piece is None:
        return False
    color = piece[0]
    pt = piece[1]
    target = board[tr][tc]
    if target is not None and target.startswith(color):
        return False
    dr = tr - fr
    dc = tc - fc
    adr = abs(dr)
    adc = abs(dc)
    if pt == 'P':
        direction = -1 if color == 'w' else 1
        start_row = 6 if color == 'w' else 1
        if dc == 0:
            if dr == direction and target is None:
                return True
            if dr == 2 * direction and fr == start_row and target is None:
                between_r = fr + direction
                if board[between_r][fc] is None:
                    return True
            return False
        if adc == 1 and dr == direction and target is not None and target[0] != color:
            return True
        return False
    if pt == 'N':
        return (adr, adc) in ((1, 2), (2, 1))
    if pt == 'B':
        if adr == adc and adr != 0:
            return is_path_clear(board, fr, fc, tr, tc)
        return False
    if pt == 'R':
        if (adr == 0 and adc != 0) or (adc == 0 and adr != 0):
            return is_path_clear(board, fr, fc, tr, tc)
        return False
    if pt == 'Q':
        if (adr == adc and adr != 0) or ((adr == 0) ^ (adc == 0)):
            return is_path_clear(board, fr, fc, tr, tc)
        return False
    if pt == 'K':
        if max(adr, adc) == 1:
            return True
        return False
    return False


def generate_moves(board: List[List[Optional[str]]], color: str) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    moves = []
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if not piece or piece[0] != color:
                continue
            for tr in range(8):
                for tc in range(8):
                    if r == tr and c == tc:
                        continue
                    if is_legal_move(board, r, c, tr, tc):
                        moves.append(((r, c), (tr, tc)))
    return moves


def make_move(board: List[List[Optional[str]]], fr: int, fc: int, tr: int, tc: int) -> Optional[str]:
    """Perform the move and return captured piece (if any). Does not handle promotion.
    Caller may handle promotion separately.
    """
    captured = board[tr][tc]
    board[tr][tc] = board[fr][fc]
    board[fr][fc] = None
    # auto-promotion to queen as in UI
    piece = board[tr][tc]
    if piece and piece[1] == 'P':
        color = piece[0]
        if (color == 'w' and tr == 0) or (color == 'b' and tr == 7):
            board[tr][tc] = f"{color}Q"
    return captured


def undo_move(board: List[List[Optional[str]]], fr: int, fc: int, tr: int, tc: int, captured: Optional[str], moved_piece: Optional[str]):
    # restore moved piece and captured piece
    board[fr][fc] = moved_piece
    board[tr][tc] = captured


def evaluate(board: List[List[Optional[str]]]) -> int:
    """Simple evaluation: sum of material (white positive).
    Returns centipawn score from White's perspective.
    """
    score = 0
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if not p:
                continue
            val = PieceValues.get(p[1], 0)
            if p[0] == 'w':
                score += val
            else:
                score -= val
    return score


def coords_to_algebraic(r: int, c: int) -> str:
    return f"{chr(ord('a') + c)}{8 - r}"


def choose_move(board: List[List[Optional[str]]], color: str, depth: int = 3) -> Optional[Tuple[str, str]]:
    """Return a move (from_sq, to_sq) for color using negamax alpha-beta.
    depth is ply depth.
    """
    best_move = None

    def negamax(node_board, node_color, ply, alpha, beta):
        if ply == 0:
            return evaluate(node_board)
        moves = generate_moves(node_board, node_color)
        if not moves:
            # no moves -> checkmate or stalemate ambiguous; use eval
            return evaluate(node_board)
        value = -9999999
        for (fr, fc), (tr, tc) in moves:
            moved_piece = node_board[fr][fc]
            captured = node_board[tr][tc]
            make_move(node_board, fr, fc, tr, tc)
            score = -negamax(node_board, 'b' if node_color == 'w' else 'w', ply - 1, -beta, -alpha)
            # undo
            undo_move(node_board, fr, fc, tr, tc, captured, moved_piece)
            if score > value:
                value = score
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return value

    moves = generate_moves(board, color)
    if not moves:
        return None
    best_score = -9999999
    alpha = -9999999
    beta = 9999999
    for (fr, fc), (tr, tc) in moves:
        moved_piece = board[fr][fc]
        captured = board[tr][tc]
        make_move(board, fr, fc, tr, tc)
        score = -negamax(board, 'b' if color == 'w' else 'w', depth - 1, -beta, -alpha)
        undo_move(board, fr, fc, tr, tc, captured, moved_piece)
        if score > best_score:
            best_score = score
            best_move = (coords_to_algebraic(fr, fc), coords_to_algebraic(tr, tc))
        alpha = max(alpha, score)
    return best_move
