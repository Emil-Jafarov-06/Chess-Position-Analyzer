from __future__ import annotations

LABELS = [
    "empty",
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king"
]

LABEL_to_INDEX = {label: index for index, label in enumerate(LABELS)}

INDEX_to_LABEL = {index: label for index, label in enumerate(LABELS)}

CLASS_COUNTS = len(LABELS)