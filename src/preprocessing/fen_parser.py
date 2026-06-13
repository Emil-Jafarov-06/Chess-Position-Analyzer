from __future__ import annotations


PIECE_LABELS = {
    "P": "white_pawn", "p": "black_pawn",
    "N": "white_knight", "n": "black_knight",
    "B": "white_bishop", "b": "black_bishop",
    "R": "white_rook", "r": "black_rook",
    "Q": "white_queen", "q": "black_queen",
    "K": "white_king", "k": "black_king"
}


def extract_piece_placement(fen: str) -> str:
    """
    Return only the visual board-layout section of a complete FEN string.
    """
    fen_parts = fen.split()
    if not fen_parts:
        raise ValueError("FEN string should not be empty!")
    return fen_parts[0]

def expand_piece_placement(piece_placement: str) -> list[list[str]]:
    """
    Convert the visible board-layout section of a FEN into an 8 × 8 grid
    containing readable class labels.
    """
    fen_rows = piece_placement.split("/")
    if len(fen_rows) != 8:
        raise ValueError("FEN string should contain exactly 8 rows!")

    board_labels: list[list[str]] = []

    for fen_row in fen_rows:
        expanded_row: list[str] = []
        for symbol in fen_row:
            if symbol.isdigit():
                number_of_empty_squares = int(symbol)
                expanded_row.extend(["empty"] * number_of_empty_squares)
            elif symbol in PIECE_LABELS:
                expanded_row.append(PIECE_LABELS[symbol])
            else:
                raise ValueError(
                    f"Unsupported character in FEN board layout: {symbol}"
                )
        if len(expanded_row) != 8:
            raise ValueError(
                "Every expanded FEN row must contain exactly 8 squares"
            )
        board_labels.append(expanded_row)

    return board_labels

def fen_to_label_grid(fen: str) -> list[list[str]]:
    """
    Convert a complete FEN string directly into an 8 × 8 label grid.
    """
    piece_placement = extract_piece_placement(fen)
    return expand_piece_placement(piece_placement)