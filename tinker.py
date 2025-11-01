import tkinter as tk
import random
from typing import List, Tuple, Optional
import time
import sys
from engine import Engine
# Named opening sequences. Each entry is a dict with a name and a list of
# (from_square, to_square) tuples in algebraic notation. You can iterate these
# and call `move_player_piece` / `move_bot_piece` accordingly.
opening_moves = [
    {"name": "King's Pawn", "moves": [("e2", "e4"), ("e7", "e5")]},
    {"name": "Sicilian Defense", "moves": [("e2", "e4"), ("c7", "c5")]},
    {"name": "French Defense", "moves": [("e2", "e4"), ("e7", "e6")]},
    {"name": "Caro-Kann", "moves": [("e2", "e4"), ("c7", "c6")]},
    {"name": "Queen's Gambit", "moves": [("d2", "d4"), ("d7", "d5"), ("c2", "c4")]},
    {"name": "Ruy Lopez", "moves": [
        ("e2", "e4"), ("e7", "e5"), ("g1", "f3"), ("b8", "c6"), ("f1", "b5"), ("a7", "a6")
    ]},
    {"name": "Italian Game", "moves": [
        ("e2", "e4"), ("e7", "e5"), ("g1", "f3"), ("b8", "c6"), ("f1", "c4"), ("f8", "c5")
    ]},
    {"name": "Scandinavian", "moves": [("e2", "e4"), ("d7", "d5")]},
    {"name": "Scholar's Mate (example)", "moves": [
        ("e2", "e4"), ("e7", "e5"), ("d1", "h5"), ("b8", "c6"), ("f1", "c4"), ("g8", "f6"), ("h5", "f7")
    ]},
]
#!/usr/bin/env python3
# tinker.py
# Simple Tkinter chessboard where one side is player-moveable (randomly black or white)
# Provides functions to move bot (immovable) pieces programmatically and to get pawn positions.


SQUARE_SIZE = 64
LIGHT_COLOR = "#F0D9B5"
DARK_COLOR = "#B58863"

# Unicode chess pieces
UNICODE = {
    "wK": "♔", "wQ": "♕", "wR": "♖", "wB": "♗", "wN": "♘", "wP": "♙",
    "bK": "♚", "bQ": "♛", "bR": "♜", "bB": "♝", "bN": "♞", "bP": "♟",
}

# Board internal representation: board[row][col], row 0 == rank 8 (top), row 7 == rank 1 (bottom)
board: List[List[Optional[str]]] = [[None for _ in range(8)] for _ in range(8)]
player_color = random.choice(["w", "b"])  # player controls this color
flip_display = player_color == "b"  # if player is black, flip board so their side at bottom
engine_color = "b" if player_color == "w" else "w"  # engine/bot controls the opposite color
current_turn = "w"  # 'w' starts

root = tk.Tk()
root.title(f"TinkerChess — You are playing: {'White' if player_color=='w' else 'Black'}")
canvas = tk.Canvas(root, width=8 * SQUARE_SIZE, height=8 * SQUARE_SIZE)
canvas.pack()

selected_sq: Optional[Tuple[int, int]] = None  # internal (row, col) of selected piece
selection_rect = None


def setup_starting_position():

    global board
    # Black major pieces (row 0), pawns (row 1)
    board[0] = ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"]
    board[1] = ["bP"] * 8
    # empty rows
    for r in range(2, 6):
        board[r] = [None] * 8
    # White pawns (row 6), major pieces (row 7)
    board[6] = ["wP"] * 8
    board[7] = ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]


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


def coords_to_algebraic(row: int, col: int) -> str:
    file = chr(ord("a") + col)
    rank = 8 - row
    return f"{file}{rank}"


def draw_board():
    """Draw the full board and pieces according to current board state."""
    global selection_rect
    canvas.delete("all")
    for r in range(8):
        for c in range(8):
            disp_r = 7 - r if flip_display else r
            x0 = c * SQUARE_SIZE
            y0 = disp_r * SQUARE_SIZE
            x1 = x0 + SQUARE_SIZE
            y1 = y0 + SQUARE_SIZE
            color = LIGHT_COLOR if (r + c) % 2 == 0 else DARK_COLOR
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=color)
            piece = board[r][c]
            if piece:
                piece_text = UNICODE.get(piece, piece)
                canvas.create_text(
                    x0 + SQUARE_SIZE / 2,
                    y0 + SQUARE_SIZE / 2,
                    text=piece_text,
                    font=("Arial", int(SQUARE_SIZE * 0.6)),
                )
    # Highlight selection if any
    if selected_sq:
        r, c = selected_sq
        disp_r = 7 - r if flip_display else r
        x0 = c * SQUARE_SIZE
        y0 = disp_r * SQUARE_SIZE
        x1 = x0 + SQUARE_SIZE
        y1 = y0 + SQUARE_SIZE
        selection_rect = canvas.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2, outline="cyan", width=3)


def pixel_to_internal_coords(x: int, y: int) -> Tuple[int, int]:
    """Map click pixel coords to internal (row,col) considering display flip."""
    col = x // SQUARE_SIZE
    disp_row = y // SQUARE_SIZE
    row = 7 - disp_row if flip_display else disp_row
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError("Click out of board")
    return row, col


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def is_path_clear(from_r: int, from_c: int, to_r: int, to_c: int) -> bool:
    """Return True if all squares between source (exclusive) and target (exclusive) are empty.

    Used for sliding pieces (rook, bishop, queen).
    """
    dr = to_r - from_r
    dc = to_c - from_c
    step_r = _sign(dr)
    step_c = _sign(dc)
    # if not a straight line or diagonal, no meaningful path to check
    if step_r == 0 and step_c == 0:
        return True
    r = from_r + step_r
    c = from_c + step_c
    while (r, c) != (to_r, to_c):
        if board[r][c] is not None:
            return False
        r += step_r
        c += step_c
    return True


def is_legal_move(from_r: int, from_c: int, to_r: int, to_c: int) -> bool:
    """Very basic legality checks for piece movement (no check detection).

    Supports pawn moves (including double-step and captures), knights, bishops,
    rooks, queens and kings. Does NOT implement castling or en-passant. Also
    does not check for leaving/being in check.
    """
    # bounds
    
    if not (0 <= from_r < 8 and 0 <= from_c < 8 and 0 <= to_r < 8 and 0 <= to_c < 8):
        return False
    piece = board[from_r][from_c]
    if piece is None:
        return False
    color = piece[0]
    piece_type = piece[1]
    target = board[to_r][to_c]
    # cannot capture own piece
    if target is not None and target.startswith(color):
        return False
    dr = to_r - from_r
    dc = to_c - from_c
    adr = abs(dr)
    adc = abs(dc)

    if piece_type == "P":
        direction = -1 if color == "w" else 1
        start_row = 6 if color == "w" else 1
        # forward move
        if dc == 0:
            # one step
            if dr == direction and target is None:
                return True
            # two steps from start
            if dr == 2 * direction and from_r == start_row and target is None:
                between_r = from_r + direction
                if board[between_r][from_c] is None:
                    return True
            return False
        # capture
        if adc == 1 and dr == direction and target is not None and target[0] != color:
            return True
        return False

    if piece_type == "N":
        return (adr, adc) in ((1, 2), (2, 1))

    if piece_type == "B":
        if adr == adc and adr != 0:
            return is_path_clear(from_r, from_c, to_r, to_c)
        return False

    if piece_type == "R":
        if (adr == 0 and adc != 0) or (adc == 0 and adr != 0):
            return is_path_clear(from_r, from_c, to_r, to_c)
        return False

    if piece_type == "Q":
        # combination of rook and bishop
        if (adr == adc and adr != 0) or ((adr == 0) ^ (adc == 0)):
            return is_path_clear(from_r, from_c, to_r, to_c)
        return False

    if piece_type == "K":
        # single-square king moves
        if max(adr, adc) == 1:
            return True
        return False

    return False


def is_square_under_pawn_attack(row: int, col: int, attacker_color: str) -> bool:
    """Return True if a pawn of attacker_color can capture a piece on (row, col).

    Uses internal (row, col) coordinates. This checks pawn capture squares only and
    does not consider en-passant.
    """
    if attacker_color not in ("w", "b"):
        raise ValueError("attacker_color must be 'w' or 'b'")
    # For a white pawn to attack (row,col), there must be a white pawn at (row+1, col-1) or (row+1, col+1)
    if attacker_color == "w":
        pr = row + 1
        if 0 <= pr < 8:
            for pc in (col - 1, col + 1):
                if 0 <= pc < 8 and board[pr][pc] == "wP":
                    return True
        return False
    else:
        # black pawns attack down the board
        pr = row - 1
        if 0 <= pr < 8:
            for pc in (col - 1, col + 1):
                if 0 <= pc < 8 and board[pr][pc] == "bP":
                    return True
        return False


def is_square_attacked(row: int, col: int, by_color: str) -> bool:
    """Return True if square (row,col) is attacked by any piece of by_color.

    This function considers basic piece attacks (pawns, knights, bishops, rooks, queens, kings).
    It does not consider special rules like en-passant attacks.
    """
    if by_color not in ("w", "b"):
        raise ValueError("by_color must be 'w' or 'b'")
    # Pawn attacks
    if is_square_under_pawn_attack(row, col, by_color):
        return True

    # Scan for other attackers
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if not piece or piece[0] != by_color:
                continue
            pt = piece[1]
            dr = row - r
            dc = col - c
            adr = abs(dr)
            adc = abs(dc)
            if pt == "N":
                if (adr, adc) in ((1, 2), (2, 1)):
                    return True
            elif pt == "B":
                if adr == adc and adr != 0 and is_path_clear(r, c, row, col):
                    return True
            elif pt == "R":
                if ((adr == 0 and adc != 0) or (adc == 0 and adr != 0)) and is_path_clear(r, c, row, col):
                    return True
            elif pt == "Q":
                if ((adr == 0 and adc != 0) or (adc == 0 and adr != 0) or (adr == adc and adr != 0)) and is_path_clear(r, c, row, col):
                    return True
            elif pt == "K":
                if max(adr, adc) == 1:
                    return True
    return False


def find_king(color: str) -> Optional[Tuple[int, int]]:
    for r in range(8):
        for c in range(8):
            if board[r][c] == f"{color}K":
                return r, c
    return None


def is_in_check(color: str) -> bool:
    """Return True if the king of `color` is currently in check."""
    king_pos = find_king(color)
    if king_pos is None:
        # No king found — treat as in check (or game over)
        return True
    kr, kc = king_pos
    attacker = "b" if color == "w" else "w"
    return is_square_attacked(kr, kc, attacker)


def is_checkmate(color: str) -> bool:
    """Return True if `color` is checkmated.

    Approach: if king not in check -> not checkmate. Otherwise, try every legal
    move for `color`; if any legal move results in king not being in check, not checkmate.
    This simulates moves and restores the board after each attempt.
    """

    if color not in ("w", "b"):
        raise ValueError("color must be 'w' or 'b'")
    if not is_in_check(color):
        return False
    # Try every legal move
    for fr in range(8):
        for fc in range(8):
            piece = board[fr][fc]
            if not piece or piece[0] != color:
                continue
            for tr in range(8):
                for tc in range(8):
                    if fr == tr and fc == tc:
                        continue
                    # skip moves that are not legal per movement rules
                    if not is_legal_move(fr, fc, tr, tc):
                        continue
                    # snapshot board
                    snapshot = [row.copy() for row in board]
                    try:
                        # perform move and postprocess (promotion)
                        perform_move_and_postprocess(fr, fc, tr, tc)
                        # after the move, if king is no longer in check -> not mate
                        if not is_in_check(color):
                            # restore and return
                            for i in range(8):
                                board[i] = snapshot[i]
                            return False
                    finally:
                        # restore board state
                        for i in range(8):
                            board[i] = snapshot[i]
    # no legal move found to escape check
    return True


def on_canvas_click(event):

    """Handle clicks to select and move player's pieces only."""
    global selected_sq
    try:
        row, col = pixel_to_internal_coords(event.x, event.y)
    except ValueError:
        return
    piece = board[row][col]
    if selected_sq is None:
        # Selecting a piece: only allow player's pieces
        if piece and piece.startswith(player_color):
            selected_sq = (row, col)
        else:
            # ignore selecting opponent piece or empty
            return
    else:
        # Try to move selected piece to clicked square
        from_r, from_c = selected_sq
        if from_r == row and from_c == col:
            # deselect
            selected_sq = None
        else:
            # Ensure destination is not occupied by player's own piece
            dest_piece = board[row][col]
            if dest_piece and dest_piece.startswith(player_color):
                # cannot capture own piece; change selection to that piece
                selected_sq = (row, col)
            else:
                # player's move should go through a restricted player move function
                move_player_piece(from_r, from_c, row, col)
                selected_sq = None
    draw_board()


def move_piece(from_r: int, from_c: int, to_r: int, to_c: int):
    """Move a piece on the internal board, but only allow engine/bot pieces.

    This function is intended for programmatic moves performed by the engine/bot.
    Player moves from the UI should use `move_player_piece` instead.
    """

    global current_turn
    piece = board[from_r][from_c]
    if piece is None:
        raise ValueError("No piece at source square")
    # only allow engine moves here
    if not piece.startswith(engine_color):
        raise ValueError("move_piece is restricted to engine/bot pieces")
    # must be engine's turn
    if current_turn != engine_color:
        raise ValueError("It's not the engine's turn")
    # validate legality
    if not is_legal_move(from_r, from_c, to_r, to_c):
        raise ValueError("Illegal move for engine piece")
    perform_move_and_postprocess(from_r, from_c, to_r, to_c)
    # switch turn
    current_turn = player_color


def move_player_piece(from_r: int, from_c: int, to_r: int, to_c: int):
    """Move a piece initiated by the player (UI). Allows only player's pieces.

    This performs the simple no-rule move (captures allowed) for the player side.
    """

    global current_turn
    piece = board[from_r][from_c]
    if piece is None:
        raise ValueError("No piece at source square")
    if not piece.startswith(player_color):
        raise ValueError("move_player_piece is restricted to the player's pieces")
    # must be player's turn
    if current_turn != player_color:
        raise ValueError("It's not the player's turn")
    # validate legality
    if not is_legal_move(from_r, from_c, to_r, to_c):
        raise ValueError("Illegal move for player piece")
    perform_move_and_postprocess(from_r, from_c, to_r, to_c)
    # switch turn
    current_turn = engine_color


def perform_move_and_postprocess(from_r: int, from_c: int, to_r: int, to_c: int):

    """Perform the physical move and handle promotions/post-move updates."""
    piece = board[from_r][from_c]
    board[to_r][to_c] = piece
    board[from_r][from_c] = None
    # Pawn promotion: promote to queen automatically
    if piece and piece[1] == "P":
        color = piece[0]
        if (color == "w" and to_r == 0) or (color == "b" and to_r == 7):
            board[to_r][to_c] = f"{color}Q"


def move_bot_piece(from_sq: str, to_sq: str):

    """
    Programmatically move an immovable (bot) piece.
    from_sq, to_sq: algebraic squares like 'e7', 'e5'
    This function will move the piece even if it belongs to player; caller should use it for bot side.
    """
    from_r, from_c = algebraic_to_coords(from_sq)
    to_r, to_c = algebraic_to_coords(to_sq)
    if board[from_r][from_c] is None:
        raise ValueError(f"No piece at {from_sq}")
    # perform move
    move_piece(from_r, from_c, to_r, to_c)
    draw_board()


def get_pawn_positions(color: str) -> List[str]:

    """Return list of algebraic squares for pawns of given color ('w' or 'b')."""
    if color not in ("w", "b"):
        raise ValueError("Color must be 'w' or 'b'")
    res = []
    for r in range(8):
        for c in range(8):
            if board[r][c] == f"{color}P":
                res.append(coords_to_algebraic(r, c))
    return res

def mov_king(color: str) :

    x, y = find_king(color)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    for dr, dc in directions:
        new_x, new_y = x + dr, y + dc
        if 0 <= new_x < 8 and 0 <= new_y < 8:
            if is_legal_move(x, y, new_x, new_y):
                if not is_square_attacked(new_x, new_y, "w" if color == "b" else "b"):
                    move_bot_piece(coords_to_algebraic(x, y), coords_to_algebraic(new_x, new_y))
                    return

def engine_move_once(depth: int = 5):
    """Ask the engine for a move at given depth and apply it once.

    Returns the move tuple (from_sq, to_sq) or None.
    """

    if is_checkmate(engine_color):
        mov_king(engine_color)
    if current_turn != engine_color:
        return None
    # copy board for engine search
    search_board = [row.copy() for row in board]

    mv = Engine.choose_move(engine_color, search_board, depth)
    print(mv)
    if not mv:
        return None
    from_sq, to_sq = mv
    line, col = algebraic_to_coords(to_sq)
    if is_square_attacked(line, col, player_color):
        engine_move_once(depth=depth + 1)
    try:
        move_bot_piece(from_sq, to_sq)
    except Exception as e:
        print("Engine move failed:", e)
        return None
    draw_board()


def engine_start(depth: int = 4, delay_ms: int = 500):
    """Start the engine loop using Tk's event loop. The engine will play when it's its turn."""

    def step():
        # stop if game over
        if is_checkmate(engine_color) or is_checkmate(player_color):
            return
        if current_turn == engine_color:
            engine_move_once(depth)
        root.after(delay_ms, step)

    root.after(delay_ms, step)
def safecheck():

    if is_checkmate(engine_color):
        print("You win")
    else:
        print("You loose")
# Bind click
canvas.bind("<Button-1>", on_canvas_click)

# Initialize and draw
setup_starting_position()
draw_board()
engine_start()
safecheck()

if __name__ == "__main__":
    # Expose some functions to the interactive namespace if needed
    root.mainloop()