from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from src.preprocessing.fen_parser import fen_to_label_grid


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POSITIONS_CSV = PROJECT_ROOT/ "data"/ "raw"/ "fens"/ "positions.csv"
PNG_ROOT = PROJECT_ROOT/ "data"/ "raw"/ "boards_png"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
SQUARE_MANIFEST_CSV = PROCESSED_ROOT/ "square_manifest.csv"
POSITION_SPLITS_CSV = PROJECT_ROOT/ "data"/ "splits"/ "position_splits.csv"
CLASS_DISTRIBUTION_CSV = PROCESSED_ROOT/ "class_distribution.csv"


# ---------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------

PIECE_STYLES = ["cburnett","merida","alpha","pirouetti"]
FILES = "abcdefgh"
RANKS = "87654321"
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42


def load_positions() -> list[dict[str, str]]:
    """
    Load the generated legal FEN positions from positions.csv.
    """
    if not POSITIONS_CSV.exists():
        raise FileNotFoundError("Could not find the file:", POSITIONS_CSV)
    with POSITIONS_CSV.open("r", encoding="utf-8") as csv_file:
        positions = list(csv.DictReader(csv_file))
    if positions:
        return positions
    else:
        raise ValueError(POSITIONS_CSV, "is empty!")


def create_position_splits(
    positions: list[dict[str, str]],
    seed: int = RANDOM_SEED,
) -> dict[str, str]:
    """
    Assign every position to train, validation, or test.
    The split happens at the POSITION level rather than the square-image
    level. This prevents data leakage: all four visual styles of the same
    chess position stay inside the same dataset split.
    """
    position_ids = [position["position_id"] for position in positions]
    random_generator = random.Random(seed)
    random_generator.shuffle(position_ids)

    total_positions = len(position_ids)
    train_end = int(total_positions * TRAIN_RATIO)
    validation_end = train_end + int(total_positions * VALIDATION_RATIO)

    split_mapping: dict[str, str] = {}

    for index, position_id in enumerate(position_ids):
        if index < train_end:
            split_mapping[position_id] = "train"
        elif index < validation_end:
            split_mapping[position_id] = "validation"
        else:
            split_mapping[position_id] = "test"
    return split_mapping


def save_position_splits(
    split_mapping: dict[str, str],
) -> None:
    """
    Save the board-level split assignments.
    """
    POSITION_SPLITS_CSV.parent.mkdir(parents=True,exist_ok=True)

    with POSITION_SPLITS_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "position_id",
                "split",
            ],
        )
        writer.writeheader()
        for position_id, split_name in sorted(
            split_mapping.items()
        ):
            writer.writerow(
                {
                    "position_id": position_id,
                    "split": split_name,
                }
            )


def create_square_manifest(
    positions: list[dict[str, str]],
    split_mapping: dict[str, str],
) -> tuple[list[dict[str, str | int]], Counter[str]]:
    """
    Create one metadata row for every chessboard square.
    No new cropped image files are saved. The square will be cropped
    dynamically from the full-board image later during model training.
    """
    manifest_rows: list[dict[str, str | int]] = []
    class_counts: Counter[str] = Counter()

    missing_images: list[Path] = []

    total_expected_boards = len(positions) * len(PIECE_STYLES)
    processed_boards = 0

    for position in positions:
        position_id = position["position_id"]
        fen = position["fen"]
        split_name = split_mapping[position_id]

        label_grid = fen_to_label_grid(fen)

        for piece_style in PIECE_STYLES:
            board_path = PNG_ROOT/ piece_style/ f"{position_id}.png"

            if not board_path.exists():
                missing_images.append(board_path)
                continue

            processed_boards += 1
            relative_board_path = board_path.relative_to(PROJECT_ROOT)

            for row in range(8):
                for column in range(8):
                    square_name = (FILES[column] + RANKS[row])
                    label = label_grid[row][column]
                    manifest_rows.append(
                        {
                            "position_id": position_id,
                            "piece_style": piece_style,
                            "board_path": str(relative_board_path),
                            "split": split_name,
                            "square_name": square_name,
                            "row": row,
                            "column": column,
                            "label": label,
                        }
                    )
                    class_counts[label] += 1

    print()
    print(f"Expected board PNG files: {total_expected_boards}")
    print(f"Found board PNG files:    {processed_boards}")
    print(f"Missing board PNG files:  {len(missing_images)}")

    if missing_images:
        print()
        print("First missing files:")

        for missing_path in missing_images[:10]:
            print(f"  {missing_path}")

        raise FileNotFoundError(
            "Some PNG board images are missing. "
            "Finish the image-generation step and run this script again."
        )
    return manifest_rows, class_counts


def save_square_manifest(
    manifest_rows: list[dict[str, str | int]],
) -> None:
    """
    Save the square-level metadata CSV used later by the PyTorch Dataset.
    """
    PROCESSED_ROOT.mkdir(parents=True,exist_ok=True,)

    with SQUARE_MANIFEST_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "position_id",
                "piece_style",
                "board_path",
                "split",
                "square_name",
                "row",
                "column",
                "label",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def save_class_distribution(
    class_counts: Counter[str],
) -> None:
    """
    Save the number of training examples available for each class.
    """
    PROCESSED_ROOT.mkdir(parents=True,exist_ok=True)

    with CLASS_DISTRIBUTION_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "label",
                "sample_count",
            ],
        )
        writer.writeheader()
        for label, sample_count in sorted(class_counts.items()):
            writer.writerow(
                {
                    "label": label,
                    "sample_count": sample_count,
                }
            )


def print_summary(
    manifest_rows: list[dict[str, str | int]],
    split_mapping: dict[str, str],
    class_counts: Counter[str],
) -> None:
    """
    Print a readable summary after preprocessing finishes.
    """
    split_position_counts = Counter(
        split_mapping.values()
    )

    split_square_counts = Counter(
        row["split"]
        for row in manifest_rows
    )

    print()
    print("Square-manifest generation finished")
    print("-----------------------------------")
    print(f"Total square samples: {len(manifest_rows)}")

    print()
    print("Position split counts:")

    for split_name, count in sorted(
        split_position_counts.items()
    ):
        print(f"  {split_name}: {count}")

    print()
    print("Square-sample split counts:")

    for split_name, count in sorted(
        split_square_counts.items()
    ):
        print(f"  {split_name}: {count}")

    print()
    print("Class distribution:")

    for label, count in sorted(
        class_counts.items()
    ):
        print(f"  {label}: {count}")

    print()
    print(f"Square manifest:     {SQUARE_MANIFEST_CSV}")
    print(f"Position splits:     {POSITION_SPLITS_CSV}")
    print(f"Class distribution:  {CLASS_DISTRIBUTION_CSV}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build square-level metadata from the downloaded "
            "chessboard PNG files."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed used for reproducible train-validation-test splits.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    positions = load_positions()

    split_mapping = create_position_splits(
        positions=positions,
        seed=arguments.seed,
    )

    save_position_splits(split_mapping)

    manifest_rows, class_counts = create_square_manifest(
        positions=positions,
        split_mapping=split_mapping,
    )

    save_square_manifest(manifest_rows)
    save_class_distribution(class_counts)

    print_summary(
        manifest_rows=manifest_rows,
        split_mapping=split_mapping,
        class_counts=class_counts,
    )


if __name__ == "__main__":
    main()