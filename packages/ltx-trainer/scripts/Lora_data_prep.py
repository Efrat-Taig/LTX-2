#!/usr/bin/env python3

"""
Orchestrate LoRA data prep: optional HF model download, prep_data_2_prep_prompts.py, then process_dataset.py.

Example:

    cd packages/ltx-trainer
    uv run python scripts/Lora_data_prep.py \\
      --local-data-dir /home/smiti-user/data/golden_skye_train \\
      --download-models

Uses HF_TOKEN from the environment if models are gated (Gemma / LTX may require accepting the license on huggingface.co).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer
from huggingface_hub import hf_hub_download, snapshot_download

app = typer.Typer(no_args_is_help=True, help="Download models, run prep_data_2, then process_dataset.")

SCRIPTS_DIR = Path(__file__).resolve().parent
PREP2 = SCRIPTS_DIR / "prep_data_2_prep_prompts.py"
PROCESS_DATASET = SCRIPTS_DIR / "process_dataset.py"

DEFAULT_LTX_REPO = "Lightricks/LTX-2"
DEFAULT_LTX_FILE = "ltx-2-19b-dev.safetensors"
DEFAULT_GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def ensure_ltx_checkpoint(
    model_path: Path,
    repo_id: str,
    filename: str,
    token: str | None,
) -> None:
    model_path = model_path.expanduser().resolve()
    if model_path.is_file():
        typer.echo(f"Using existing LTX checkpoint: {model_path}")
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading {filename} from {repo_id} (very large; can take a long time)...")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(model_path.parent),
        token=token,
    )
    expected = model_path.parent / filename
    if not model_path.is_file() and expected.is_file() and expected != model_path:
        expected.rename(model_path)
    if not model_path.is_file():
        msg = f"Expected checkpoint at {model_path} after download"
        raise RuntimeError(msg)


def ensure_gemma_dir(model_dir: Path, repo_id: str, token: str | None) -> None:
    model_dir = model_dir.expanduser().resolve()
    marker = model_dir / "config.json"
    if marker.is_file():
        typer.echo(f"Using existing Gemma directory: {model_dir}")
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Downloading {repo_id} to {model_dir} (large; can take a long time)...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        token=token,
        local_dir_use_symlinks=False,
    )
    if not (model_dir / "config.json").is_file():
        msg = f"Download finished but config.json not found under {model_dir}"
        raise RuntimeError(msg)


def _require_local_models(model_path: Path, text_encoder_path: Path) -> None:
    """Exit with a clear message if checkpoints are missing (e.g. user passed --no-download-models)."""
    if not model_path.is_file():
        typer.echo(
            f"LTX checkpoint not found: {model_path}\n"
            "  Fix: run without --no-download-models, or place ltx-2-19b-dev.safetensors there, "
            "or pass --model-path to an existing .safetensors file.",
            err=True,
        )
        raise typer.Exit(code=1)
    gemma_cfg = text_encoder_path / "config.json"
    if not text_encoder_path.is_dir() or not gemma_cfg.is_file():
        typer.echo(
            f"Gemma model directory missing or incomplete: {text_encoder_path}\n"
            "  Expected a folder containing config.json (Hugging Face Gemma layout).\n"
            "  Fix: omit --no-download-models so this script can download from Hugging Face, "
            "or download manually:\n"
            "    huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized "
            f"--local-dir {text_encoder_path}\n"
            "  Gated models need HF_TOKEN and accepting the license on huggingface.co.",
            err=True,
        )
        raise typer.Exit(code=1)


def _run(cmd: list[str]) -> None:
    typer.echo(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(SCRIPTS_DIR))


@app.command()
def main(
    local_data_dir: Path = typer.Option(
        Path("/home/smiti-user/data/golden_skye_train"),
        "--local-data-dir",
        "-o",
        help="Where prep_data_2 writes videos + dataset.json (Step 2 local folder)",
    ),
    parquet: str = typer.Option(
        "gs://video_gen_dataset/TinyStories/data_sets/golden_skye/videos/skye_golden_dataset_v2.parquet",
        "--parquet",
        help="Parquet manifest for prep_data_2",
    ),
    strip_blob_prefix: str = typer.Option(
        "TinyStories/data_sets/golden_skye/videos/",
        "--strip-blob-prefix",
        help="See prep_data_2_prep_prompts.py",
    ),
    video_column: str = typer.Option("output_video_path", "--video-column"),
    caption_column: str = typer.Option("scene_caption", "--caption-column"),
    dedupe_videos: bool = typer.Option(True, "--dedupe-videos/--no-dedupe-videos"),
    prep_workers: int = typer.Option(16, "--prep-workers", "-j", min=1),
    download_separate_audio: bool = typer.Option(
        True,
        "--download-separate-audio/--no-download-separate-audio",
        help="Also fetch output_audio_path sidecars (see prep_data_2)",
    ),
    audio_column: str = typer.Option("output_audio_path", "--audio-column"),
    skip_prep2: bool = typer.Option(
        False,
        "--skip-prep2",
        help="Skip prep_data_2 if dataset.json already exists and you only want preprocess",
    ),
    resolution_buckets: str = typer.Option(
        "960x544x49",
        "--resolution-buckets",
        help="process_dataset resolution buckets",
    ),
    model_path: Path = typer.Option(
        Path("/home/smiti-user/models/ltx-2-19b-dev.safetensors"),
        "--model-path",
        help="Local path for LTX-2 .safetensors (downloaded if missing and --download-models)",
    ),
    text_encoder_path: Path = typer.Option(
        Path("/home/smiti-user/models/gemma-3-12b-it-qat-q4_0-unquantized"),
        "--text-encoder-path",
        help="Local directory for Gemma (downloaded if missing and --download-models)",
    ),
    download_models: bool = typer.Option(
        True,
        "--download-models/--no-download-models",
        help="Download LTX + Gemma when missing; if disabled, paths must already exist (validated before preprocess)",
    ),
    ltx_repo: str = typer.Option(DEFAULT_LTX_REPO, "--ltx-repo"),
    ltx_filename: str = typer.Option(DEFAULT_LTX_FILE, "--ltx-filename"),
    gemma_repo: str = typer.Option(DEFAULT_GEMMA_REPO, "--gemma-repo"),
    process_batch_size: int = typer.Option(1, "--process-batch-size", min=1),
    device: str = typer.Option("cuda", "--device"),
    with_audio: bool = typer.Option(True, "--with-audio/--no-with-audio"),
    decode: bool = typer.Option(False, "--decode", help="VAE-decode latents for sanity check"),
    load_text_encoder_in_8bit: bool = typer.Option(
        False,
        "--load-text-encoder-in-8bit",
        help="Lower VRAM during caption embedding (bitsandbytes)",
    ),
    precomputed_output_dir: str | None = typer.Option(
        None,
        "--precomputed-output-dir",
        help="Optional; passed to process_dataset --output-dir (default: dataset_dir/.precomputed)",
    ),
) -> None:
    """Run prep_data_2_prep_prompts.py, then process_dataset.py; optionally download checkpoints from Hugging Face."""
    token = _hf_token()
    local_data_dir = local_data_dir.expanduser().resolve()
    dataset_json = local_data_dir / "dataset.json"

    model_path = model_path.expanduser().resolve()
    text_encoder_path = text_encoder_path.expanduser().resolve()

    if download_models:
        ensure_ltx_checkpoint(model_path, ltx_repo, ltx_filename, token)
        ensure_gemma_dir(text_encoder_path, gemma_repo, token)
    else:
        _require_local_models(model_path, text_encoder_path)

    if not skip_prep2:
        cmd: list[str] = [
            sys.executable,
            str(PREP2),
            "--parquet",
            parquet,
            "--output-dir",
            str(local_data_dir),
            "--strip-blob-prefix",
            strip_blob_prefix,
            "--video-column",
            video_column,
            "--caption-column",
            caption_column,
            "--dedupe-videos" if dedupe_videos else "--no-dedupe-videos",
            "--workers",
            str(prep_workers),
        ]
        if download_separate_audio:
            cmd.append("--download-separate-audio")
        else:
            cmd.append("--no-download-separate-audio")
        cmd.extend(["--audio-column", audio_column])
        _run(cmd)
    else:
        typer.echo("--skip-prep2: not running prep_data_2_prep_prompts.py")

    if not dataset_json.is_file():
        typer.echo(f"Missing {dataset_json}; cannot run process_dataset.", err=True)
        raise typer.Exit(code=1)

    pcmd: list[str] = [
        sys.executable,
        str(PROCESS_DATASET),
        str(dataset_json),
        "--resolution-buckets",
        resolution_buckets,
        "--model-path",
        str(model_path),
        "--text-encoder-path",
        str(text_encoder_path),
        "--batch-size",
        str(process_batch_size),
        "--device",
        device,
    ]
    if with_audio:
        pcmd.append("--with-audio")
    if decode:
        pcmd.append("--decode")
    if load_text_encoder_in_8bit:
        pcmd.append("--load-text-encoder-in-8bit")
    if precomputed_output_dir:
        pcmd.extend(["--output-dir", precomputed_output_dir])

    _run(pcmd)
    typer.echo("Done. Precomputed data is next to your dataset (typically .precomputed/).")


if __name__ == "__main__":
    app()
