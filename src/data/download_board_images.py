from __future__ import annotations

import argparse
import csv
import random
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlencode

import requests
from PIL import Image, UnidentifiedImageError


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POSITIONS_CSV = PROJECT_ROOT / "data" / "raw" / "fens" / "positions.csv"

GIF_ROOT = PROJECT_ROOT / "data" / "raw" / "boards_gif"
PNG_ROOT = PROJECT_ROOT / "data" / "raw" / "boards_png"

MANIFEST_CSV = PROJECT_ROOT / "data" / "raw" / "image_manifest.csv"


# ---------------------------------------------------------
# Lichess rendering configuration
# ---------------------------------------------------------

LICHESS_IMAGE_URL = "http://localhost:6175/image.gif"

DEFAULT_PIECE_STYLES = [
    "cburnett",
    "merida",
    "alpha",
    "pirouetti",
]

BOARD_THEME = "brown"
BOARD_ORIENTATION = "white"


# ---------------------------------------------------------
# Download-rate configuration
# ---------------------------------------------------------

# Create a fresh HTTP session after this number of actual requests.
REQUESTS_PER_BATCH = 500

# Pause after every batch before creating the next session.
BATCH_PAUSE_SECONDS = 0

# Always pause for at least one minute after receiving HTTP 429.
MINIMUM_RATE_LIMIT_WAIT_SECONDS = 2


def create_session() -> requests.Session:
    """
    Create a reusable HTTP session.

    The session is recreated after each batch for clean connection handling.
    Recreating a session does not bypass a server-side rate limit.
    """
    session = requests.Session()

    session.headers.update(
        {
            "User-Agent": (
                "ChessPositionAnalyzerDatasetCollector/1.0 "
                "(educational student project)"
            )
        }
    )

    return session


def create_lichess_url(fen: str, piece_style: str) -> str:
    """
    Create a local renderer URL for one FEN position and one piece style.
    """
    parameters = {
        "fen": fen,
        "orientation": BOARD_ORIENTATION,
        "theme": BOARD_THEME,
        "piece": piece_style,
    }

    return f"{LICHESS_IMAGE_URL}?{urlencode(parameters)}"


def download_and_convert_image(
    session: requests.Session,
    image_url: str,
    gif_path: Path,
    png_path: Path,
    timeout_seconds: int = 30,
    max_attempts: int = 5,
) -> None:
    """
    Download one GIF image and convert it into PNG format.

    If Lichess temporarily rate-limits the requests, pause and retry
    the same image rather than immediately continuing to the next one.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(
                image_url,
                timeout=timeout_seconds,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                wait_seconds = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else MINIMUM_RATE_LIMIT_WAIT_SECONDS
                )

                wait_seconds = max(
                    wait_seconds,
                    MINIMUM_RATE_LIMIT_WAIT_SECONDS,
                )

                print()
                print(
                    f"Rate limit reached. Waiting {wait_seconds} seconds "
                    f"before retrying attempt {attempt}/{max_attempts}..."
                )
                print()

                time.sleep(wait_seconds)
                continue

            response.raise_for_status()

            gif_path.write_bytes(response.content)

            with Image.open(BytesIO(response.content)) as image:
                image.seek(0)
                image.convert("RGB").save(
                    png_path,
                    format="PNG",
                )

            return

        except requests.RequestException as error:
            if attempt == max_attempts:
                raise

            wait_seconds = min(60, 2 ** attempt)

            print()
            print(
                f"Temporary request error: {error}. "
                f"Waiting {wait_seconds} seconds before retrying..."
            )
            print()

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Could not download image after {max_attempts} attempts: {image_url}"
    )


def save_manifest(manifest_rows: list[dict[str, str]]) -> None:
    """
    Save the metadata collected so far.

    This is also called when the script is stopped manually, so the
    completed progress is not lost.
    """
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "position_id",
                "piece_style",
                "theme",
                "orientation",
                "fen",
                "gif_path",
                "png_path",
                "status",
                "url",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)


def download_dataset(
    limit: int | None = None,
    delay_seconds: float = 0.0,
    piece_styles: list[str] | None = None,
) -> None:
    """
    Download each FEN position using several Lichess piece styles.

    Existing PNG images are skipped. Therefore, the script can be
    stopped and restarted safely.
    """
    if not POSITIONS_CSV.exists():
        raise FileNotFoundError(
            f"Could not find the FEN dataset: {POSITIONS_CSV}"
        )

    selected_styles = piece_styles or DEFAULT_PIECE_STYLES

    with POSITIONS_CSV.open("r", encoding="utf-8") as csv_file:
        positions = list(csv.DictReader(csv_file))

    if limit is not None:
        positions = positions[:limit]

    GIF_ROOT.mkdir(parents=True, exist_ok=True)
    PNG_ROOT.mkdir(parents=True, exist_ok=True)

    session = create_session()

    manifest_rows: list[dict[str, str]] = []

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0

    requests_in_current_batch = 0

    total_images = len(positions) * len(selected_styles)
    current_image = 0

    interrupted = False

    try:
        for position in positions:
            position_id = position["position_id"]
            fen = position["fen"]

            for piece_style in selected_styles:
                current_image += 1

                gif_directory = GIF_ROOT / piece_style
                png_directory = PNG_ROOT / piece_style

                gif_directory.mkdir(parents=True, exist_ok=True)
                png_directory.mkdir(parents=True, exist_ok=True)

                gif_path = gif_directory / f"{position_id}.gif"
                png_path = png_directory / f"{position_id}.png"

                image_url = create_lichess_url(
                    fen=fen,
                    piece_style=piece_style,
                )

                status = "downloaded"
                request_was_made = False

                if png_path.exists():
                    skipped_count += 1
                    status = "skipped"

                    print(
                        f"[{current_image}/{total_images}] "
                        f"Skipped existing file: "
                        f"{piece_style}/{png_path.name}"
                    )

                else:
                    request_was_made = True

                    try:
                        download_and_convert_image(
                            session=session,
                            image_url=image_url,
                            gif_path=gif_path,
                            png_path=png_path,
                        )

                        downloaded_count += 1

                        print(
                            f"[{current_image}/{total_images}] "
                            f"Downloaded: {piece_style}/{png_path.name}"
                        )

                    except (
                        requests.RequestException,
                        UnidentifiedImageError,
                        OSError,
                        RuntimeError,
                    ) as error:
                        failed_count += 1
                        status = "failed"

                        print(
                            f"[{current_image}/{total_images}] "
                            f"Failed: {piece_style}/{position_id} -> {error}"
                        )

                manifest_rows.append(
                    {
                        "position_id": position_id,
                        "piece_style": piece_style,
                        "theme": BOARD_THEME,
                        "orientation": BOARD_ORIENTATION,
                        "fen": fen,
                        "gif_path": str(
                            gif_path.relative_to(PROJECT_ROOT)
                        ),
                        "png_path": str(
                            png_path.relative_to(PROJECT_ROOT)
                        ),
                        "status": status,
                        "url": image_url,
                    }
                )

                if request_was_made:
                    requests_in_current_batch += 1

                    # Small random variation prevents perfectly timed requests.
                    time.sleep(
                        delay_seconds + random.uniform(0.0, 0.5)
                    )

                    if requests_in_current_batch >= REQUESTS_PER_BATCH:
                        print()
                        print(
                            f"Completed {REQUESTS_PER_BATCH} requests. "
                            f"Waiting {BATCH_PAUSE_SECONDS} seconds "
                            f"before starting the next batch..."
                        )
                        print()

                        session.close()

                        time.sleep(BATCH_PAUSE_SECONDS)

                        session = create_session()
                        requests_in_current_batch = 0

    except KeyboardInterrupt:
        interrupted = True

        print()
        print("Download stopped manually. Saving current progress...")

    finally:
        session.close()
        save_manifest(manifest_rows)

    print()
    print("Dataset collection summary")
    print("--------------------------")
    print(f"Downloaded: {downloaded_count}")
    print(f"Skipped:    {skipped_count}")
    print(f"Failed:     {failed_count}")
    print(f"Manifest:   {MANIFEST_CSV}")

    if interrupted:
        print()
        print(
            "You can run the same command again later. "
            "Existing PNG files will be skipped automatically."
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Lichess chessboard images and convert GIF files "
            "into PNG files."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download only the first N FEN positions while testing.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between individual image requests in seconds.",
    )

    parser.add_argument(
        "--styles",
        nargs="+",
        default=DEFAULT_PIECE_STYLES,
        help="Piece styles to download.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    download_dataset(
        limit=arguments.limit,
        delay_seconds=arguments.delay,
        piece_styles=arguments.styles,
    )


if __name__ == "__main__":
    main()