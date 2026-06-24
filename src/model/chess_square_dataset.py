from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.model.labels import LABEL_to_INDEX

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SQUARE_MANIFEST_CSV = (PROJECT_ROOT / "data" / "processed" / "square_manifest.csv")

def build_square_transform() -> transforms.Compose:
    """
    Build the image transformation pipeline used before sending
    a square image to the CNN.
    """
    return transforms.Compose(
        [
            transforms.Resize((64, 64)), # each square is 64x64
            transforms.ToTensor(), # from png into tensor
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )

class ChessSquareDataset(Dataset):
    """
    PyTorch Dataset for square-level chess piece classification.

    Each row in square_manifest.csv tells the dataset:
        - which full-board image to open
        - which row and column to crop
        - what the correct label is
    """

    def __init__(
        self,
        manifest_path: Path = SQUARE_MANIFEST_CSV,
        split: str = "train",
        transform: transforms.Compose | None = None,
    ) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Could not find square manifest: {manifest_path}"
            )

        if split not in {"train", "validation", "test"}:
            raise ValueError(
                "split must be one of: train, validation, test"
            )

        manifest = pd.read_csv(manifest_path)
        manifest = manifest[manifest["split"] == split].reset_index(
            drop=True
        )

        if manifest.empty:
            raise ValueError(
                f"No rows found for split: {split}"
            )

        self.manifest = manifest
        self.transform = transform or build_square_transform()

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row_data = self.manifest.iloc[index]

        board_path = PROJECT_ROOT / row_data["board_path"]

        board_image = Image.open(board_path).convert("RGB")

        board_width, board_height = board_image.size

        if board_width != board_height:
            raise ValueError(
                f"Expected square board image, got {board_width}x{board_height}: "
                f"{board_path}"
            )

        square_size = board_width // 8

        row = int(row_data["row"])
        column = int(row_data["column"])

        left = column * square_size
        upper = row * square_size
        right = left + square_size
        lower = upper + square_size

        square_image = board_image.crop(
            (left, upper, right, lower)
        )

        image_tensor = self.transform(square_image)

        label_name = row_data["label"]
        label_index = LABEL_to_INDEX[label_name]

        label_tensor = torch.tensor(
            label_index,
            dtype=torch.long,
        )

        return image_tensor, label_tensor