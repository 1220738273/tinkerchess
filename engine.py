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
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

PieceValues = {
    'P': 100,
    'N': 320,
    'B': 330,
    'R': 500,
    'Q': 900,
    'K': 20000,
}

class PositionalNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8*8*12, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Engine:
    """Simple configurable engine instance.

    Holds piece values so we can tune them during training.
    """
    def __init__(self, piece_values: Optional[dict] = None):
        self.piece_values = piece_values.copy() if piece_values else PieceValues.copy()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = PositionalNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()


    @staticmethod
    def algebraic_to_coords(sq: str) -> Tuple[int, int]:
        """Convert algebraic like 'e2' to internal (row, col)."""
        if len(sq) != 2:
            raise ValueError("Square must be in form like 'e2'")
        file = sq[0].lower()
        rank = int(sq[1])
        col = ord(file) - ord("a")
        row = 8 - rank
        if not (0 <= row < 8 and 0 <= col < 8):
            raise ValueError("Square out of range")
        return row, col

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
    def choose_move(board: List[List[Optional[str]]], color: str, depth: int = 3) -> Optional[Tuple[str, str]]:
        INF = 1e9

        def negamax(bd, col, d, alpha, beta, ply=0):
            # deep quiescence: prioritize checks and captures
            moves = generate_moves(bd, col)
            if d == 0 or not moves:
                return Engine.evaluate(bd, col)

            best_val = -INF
            next_col = 'b' if col == 'w' else 'w'

            # sort moves: check/capture first
            def move_score(move):
                fr, fc = move[0]
                tr, tc = move[1]
                target = bd[tr][tc]
                score = 0
                if target:  # capture
                    score += PieceValues.get(target[1], 0)
                make_move(bd, fr, fc, tr, tc)
                if is_king_in_check(bd, next_col):
                    score += 5000  # check bonus
                undo_move(bd, fr, fc, tr, tc, target, bd[fr][fc])
                return score

            moves.sort(key=move_score, reverse=True)

            for (fr, fc), (tr, tc) in moves:
                moved_piece = bd[fr][fc]
                captured = bd[tr][tc]
                make_move(bd, fr, fc, tr, tc)
                val = -negamax(bd, next_col, d - 1, -beta, -alpha, ply + 1)
                undo_move(bd, fr, fc, tr, tc, captured, moved_piece)
                if val > best_val:
                    best_val = val
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return best_val

        moves = generate_moves(board, color)
        if not moves:
            return None

        best_score = -INF
        best_move = random.choice(moves)
        next_col = 'b' if color == 'w' else 'w'

        for (fr, fc), (tr, tc) in moves:
            moved_piece = board[fr][fc]
            captured = board[tr][tc]
            make_move(board, fr, fc, tr, tc)
            score = -negamax(board, next_col, depth - 1, -INF, INF)
            undo_move(board, fr, fc, tr, tc, captured, moved_piece)
            if score > best_score:
                best_score = score
                best_move = (fr, fc, tr, tc)

        fr, fc, tr, tc = best_move
        return coords_to_algebraic(fr, fc), coords_to_algebraic(tr, tc)


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
            fr_r, fr_c = Engine.algebraic_to_coords(fr)
            tr_r, tr_c = Engine.algebraic_to_coords(to)
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
    def board_to_tensor(self, board: List[List[Optional[str]]]) -> torch.Tensor:
        mapping = ['wP','wN','wB','wR','wQ','wK','bP','bN','bB','bR','bQ','bK']
        tensor = torch.zeros(12,8,8, dtype=torch.float32)
        for r in range(8):
            for c in range(8):
                piece = board[r][c]
                if piece:
                    idx = mapping.index(piece)
                    tensor[idx,r,c] = 1.0
        return tensor.unsqueeze(0).to(self.device)
    def nn_evaluate(self, board: List[List[Optional[str]]]) -> float:
        """Neural net evaluation inside the engine."""
        tensor = self.board_to_tensor(board)
        with torch.no_grad():
            return self.nn(tensor).item()
        



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
    """
    Generate all pseudo-legal moves for color.
    - Accepts color as 'w'|'b' or 'white'|'black' (case-insensitive).
    - Validates board shape and piece formats and raises helpful errors if something is wrong.
    """
    # --- normalize color ---
    if not isinstance(color, str):
        raise ValueError(f"color must be a string 'w' or 'b', got: {color!r} (type {type(color)})")
    lc = color.lower()
    if lc in ('w', 'white'):
        color = 'w'
    elif lc in ('b', 'black'):
        color = 'b'
    else:
        raise ValueError(f"color must be 'w'/'white' or 'b'/'black', got: {color!r}")

    # --- validate board shape ---
    if not (isinstance(board, list) and len(board) == 8):
        raise ValueError(f"board must be a list of 8 rows, got type {type(board)} with length {len(board) if hasattr(board,'__len__') else 'n/a'}")
    for i, row in enumerate(board):
        if not (isinstance(row, list) and len(row) == 8):
            raise ValueError(f"board row {i} must be a list of length 8, got {type(row)} length {len(row) if hasattr(row,'__len__') else 'n/a'} -> {row!r}")

    moves: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []

    directions = {
        'N': [(-2, -1), (-1, -2), (-2, 1), (-1, 2), (1, -2), (2, -1), (1, 2), (2, 1)],
        'B': [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        'R': [(-1, 0), (1, 0), (0, -1), (0, 1)],
        'Q': [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)],
        'K': [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)],
    }

    VALID_PIECE_CHARS = set(('P', 'N', 'B', 'R', 'Q', 'K'))

    for r in range(8):
        for c in range(8):
            piece = board[r][c]

            # skip empty squares quickly
            if piece is None:
                continue

            # piece must be a 2-char string like 'wP' or 'bK'
            if not isinstance(piece, str):
                raise ValueError(f"board[{r}][{c}] is not None or str: {piece!r} (type {type(piece)})")
            s = piece.strip()
            if len(s) < 2:
                raise ValueError(f"board[{r}][{c}] has invalid piece string: {piece!r}")
            # allow forms like 'wP' or 'Wp' or ' wP '
            p_color = s[0].lower()
            p_type = s[1].upper()

            if p_color not in ('w', 'b'):
                raise ValueError(f"board[{r}][{c}] has invalid color char: {s!r}")
            if p_type not in VALID_PIECE_CHARS:
                raise ValueError(f"board[{r}][{c}] has invalid piece type: {s!r}")

            # only process pieces of requested color
            if p_color != color:
                continue

            pt = p_type  # now guaranteed to be 'P','N','B','R','Q','K'

            # Pawn logic
            if pt == 'P':
                direction = -1 if color == 'w' else 1
                start_row = 6 if color == 'w' else 1

                tr = r + direction
                # single step
                if 0 <= tr < 8 and board[tr][c] is None:
                    moves.append(((r, c), (tr, c)))
                    # double step: ensure both squares empty and within bounds
                    tr2 = r + 2 * direction
                    if r == start_row and 0 <= tr2 < 8 and board[tr2][c] is None and board[tr][c] is None:
                        moves.append(((r, c), (tr2, c)))

                # captures
                for dc in (-1, 1):
                    tc = c + dc
                    if 0 <= tc < 8 and 0 <= tr < 8:
                        target = board[tr][tc]
                        if target is not None:
                            if not isinstance(target, str):
                                raise ValueError(f"board[{tr}][{tc}] unexpected type: {target!r}")
                            targ_color = target.strip()[0].lower()
                            if targ_color != color:
                                moves.append(((r, c), (tr, tc)))

            # Knight or King (fixed offsets)
            elif pt in ('N', 'K'):
                for dr, dc in directions[pt]:
                    tr, tc = r + dr, c + dc
                    if 0 <= tr < 8 and 0 <= tc < 8:
                        target = board[tr][tc]
                        if target is None:
                            moves.append(((r, c), (tr, tc)))
                        else:
                            if not isinstance(target, str):
                                raise ValueError(f"board[{tr}][{tc}] unexpected type: {target!r}")
                            if target.strip()[0].lower() != color:
                                moves.append(((r, c), (tr, tc)))

            # Sliding pieces: Bishop, Rook, Queen
            elif pt in ('B', 'R', 'Q'):
                for dr, dc in directions[pt]:
                    tr, tc = r + dr, c + dc
                    while 0 <= tr < 8 and 0 <= tc < 8:
                        target = board[tr][tc]
                        if target is None:
                            moves.append(((r, c), (tr, tc)))
                        else:
                            if not isinstance(target, str):
                                raise ValueError(f"board[{tr}][{tc}] unexpected type: {target!r}")
                            if target.strip()[0].lower() != color:
                                moves.append(((r, c), (tr, tc)))
                            break  # blocked
                        tr += dr
                        tc += dc

            else:
                # should not happen due to earlier validation, but keep safe
                raise ValueError(f"Unhandled piece type at board[{r}][{c}]: {piece!r}")

    return moves



def make_move(board: List[List[Optional[str]]], fr: int, fc: int, tr: int, tc: int) -> Optional[str]:
    """Perform the move and return captured piece (if any). Auto-promotes pawns to queen."""
    captured = board[tr][tc]
    board[tr][tc] = board[fr][fc]
    board[fr][fc] = None
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



def evaluate(self, board: List[List[Optional[str]]], color: str) -> float:
    """Combine classic material eval + neural net."""
    mat_score = sum(
        self.piece_values.get(p[1], 0) if p[0] == color else -self.piece_values.get(p[1], 0)
        for row in board for p in row if p
    )
    nn_score = self.nn_evaluate(board)
    return 0.6*mat_score + 0.4*nn_score
def coords_to_algebraic(r: int, c: int) -> str:
    file = chr(ord('a') + c)
    rank = str(8 - r)
    return file + rank
def is_king_in_check(board: List[List[Optional[str]]], color: str) -> bool:
    # find king
    king_pos = None
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p == f"{color}K":
                king_pos = (r, c)
                break
        if king_pos:
            break
    if not king_pos:
        return False  # no king found, treat as not in check

    kr, kc = king_pos
    opponent = 'b' if color == 'w' else 'w'
    moves = generate_moves(board, opponent)
    for (fr, fc), (tr, tc) in moves:
        if (tr, tc) == (kr, kc):
            return True
    return False