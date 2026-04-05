#!/usr/bin/env python3
"""
End-to-end data prep for LTX-2 LoRA training from a long (or single) video.

Follows the workflow in packages/ltx-trainer/docs/dataset-preparation.md:

1. (Optional) Split into scenes — split_scenes.py
2. **Caption clips (on by default)** — caption_videos.py → writes `dataset.json` prompts
3. Preprocess (latents + text embeddings) — process_dataset.py

Use `--no-caption` only if you already have a manifest (`--dataset-json`). To caption but skip GPU preprocess: `--no-preprocess`.

Example (from repo root, after `uv sync`):

    uv run python packages/ltx-trainer/scripts/prep_long_video_to_lora_dataset.py \\
      --work-dir ./data/paw_patrol_s01e01

`--model-path` defaults to `<LTX-2 repo>/models/ltx-2.3-22b-dev.safetensors` (place the HF download there).
`--text-encoder-path` defaults to `<LTX-2 repo>/models/gemma-3-12b-it-qat-q4_0-unquantized`
(HF: google/gemma-3-12b-it-qat-q4_0-unquantized). Override if your checkout lives elsewhere.

Default source is the Paw Patrol S01E01 clip on GCS. Override with --source.

Requires: gsutil on PATH for gs:// inputs; GPU recommended for captioning and preprocessing.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help="GCS/local video → LoRA-ready preprocessed dataset (LTX-2 trainer).")


def _trainer_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ltx_repo_root() -> Path:
    """packages/ltx-trainer/scripts/this_file.py -> LTX-2 monorepo root."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _default_text_encoder_path() -> str:
    return str(_ltx_repo_root() / "models" / "gemma-3-12b-it-qat-q4_0-unquantized")


def _default_model_path() -> str:
    """LTX-2 checkpoint beside Gemma under <repo>/models/ (see LTX-2 README / Hugging Face)."""
    return str(_ltx_repo_root() / "models" / "ltx-2.3-22b-dev.safetensors")


def _is_gs_uri(s: str) -> bool:
    return s.strip().startswith("gs://")


def gsutil_cp(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gsutil", "cp", src, str(dst)], check=True)


def run_py_script(script_name: str, args: list[str], *, cwd: Path) -> None:
    script = cwd / "scripts" / script_name
    cmd = [sys.executable, str(script), *args]
    typer.echo(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


@app.command()
def main(
    source: str = typer.Option(
        "gs://video_gen_dataset/raw/paw_patrol/season_1/compressed_videos/PawPatrol_S01_E01_A.mp4",
        "--source",
        "-s",
        help="Local path or gs:// URI to the input video",
    ),
    work_dir: Path = typer.Option(
        Path("./lora_prep_from_video"),
        "--work-dir",
        "-w",
        help="Local working directory (raw video, scenes, dataset.json, .precomputed)",
    ),
    split_scenes: bool = typer.Option(
        True,
        "--split/--no-split",
        help="Split long video into scenes before captioning (recommended for long-form)",
    ),
    filter_shorter_than: str = typer.Option(
        "5s",
        "--filter-shorter-than",
        help="Drop scenes shorter than this (passed to split_scenes.py)",
    ),
    caption: bool = typer.Option(
        True,
        "--caption/--no-caption",
        help="Run automatic captioning via caption_videos.py (default: on). "
        "Use --no-caption if you supply --dataset-json with captions already.",
    ),
    captioner_type: str = typer.Option(
        "qwen_omni",
        "--captioner-type",
        "-c",
        help="qwen_omni (local) or gemini_flash (API)",
    ),
    caption_use_8bit: bool = typer.Option(
        False,
        "--caption-use-8bit",
        help="8-bit captioning model (lower VRAM)",
    ),
    preprocess: bool = typer.Option(
        True,
        "--preprocess/--no-preprocess",
        help="Run process_dataset.py (VAE latents + Gemma embeddings)",
    ),
    model_path: str = typer.Option(
        _default_model_path(),
        "--model-path",
        help="LTX-2 .safetensors checkpoint (default: <repo>/models/ltx-2.3-22b-dev.safetensors)",
    ),
    text_encoder_path: str = typer.Option(
        _default_text_encoder_path(),
        "--text-encoder-path",
        help=(
            "Gemma 3 model directory (default: <repo>/models/gemma-3-12b-it-qat-q4_0-unquantized; "
            "HF: google/gemma-3-12b-it-qat-q4_0-unquantized)"
        ),
    ),
    resolution_buckets: str = typer.Option(
        "960x544x49",
        "--resolution-buckets",
        help='e.g. "960x544x49" — see dataset-preparation.md',
    ),
    preprocess_batch_size: int = typer.Option(1, "--preprocess-batch-size", min=1),
    device: str = typer.Option("cuda", "--device", help="cuda or cpu for preprocessing"),
    vae_tiling: bool = typer.Option(False, "--vae-tiling", help="VAE tiling for large resolutions"),
    with_audio: bool = typer.Option(
        False,
        "--with-audio/--no-audio-preprocess",
        help="Encode audio latents for audio–video LoRA training",
    ),
    load_text_encoder_in_8bit: bool = typer.Option(
        False,
        "--load-text-encoder-in-8bit",
        help="8-bit Gemma during preprocessing (saves VRAM)",
    ),
    lora_trigger: str | None = typer.Option(
        None,
        "--lora-trigger",
        help="Optional trigger token prepended to every caption during preprocessing",
    ),
    decode: bool = typer.Option(
        False,
        "--decode",
        help="Decode latents after preprocessing for visual verification",
    ),
    dataset_json: Path | None = typer.Option(
        None,
        "--dataset-json",
        help="Use this dataset.json instead of captioning (implies --no-caption)",
    ),
) -> None:
    """Download (if gs://), optionally split scenes, caption, and preprocess for LoRA training."""
    work_dir = work_dir.resolve()
    raw_dir = work_dir / "raw"
    scenes_dir = work_dir / "scenes"
    dataset_path = dataset_json.resolve() if dataset_json else (work_dir / "dataset.json")

    if dataset_json is not None:
        caption = False
        if not dataset_path.is_file():
            typer.echo(f"--dataset-json not found: {dataset_path}", err=True)
            raise typer.Exit(code=1)

    trainer = _trainer_root()

    if dataset_json is None:
        stem = Path(source.split("?", 1)[0]).stem
        if not stem:
            stem = "input"

        local_video = raw_dir / f"{stem}.mp4"

        typer.echo(f"Work directory: {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1) Acquire local video
        if _is_gs_uri(source):
            typer.echo(f"Downloading {source} -> {local_video}")
            gsutil_cp(source, local_video)
        else:
            p = Path(source).expanduser().resolve()
            if not p.is_file():
                typer.echo(f"Video not found: {p}", err=True)
                raise typer.Exit(code=1)
            raw_dir.mkdir(parents=True, exist_ok=True)
            if p.resolve() != local_video.resolve():
                shutil.copy2(p, local_video)
            typer.echo(f"Using local video: {local_video}")

        media_for_caption: Path

        # 2) Optional scene split
        if split_scenes:
            if scenes_dir.exists():
                shutil.rmtree(scenes_dir)
            scenes_dir.mkdir(parents=True)
            typer.echo("Splitting scenes...")
            run_py_script(
                "split_scenes.py",
                [
                    str(local_video),
                    str(scenes_dir),
                    "--filter-shorter-than",
                    filter_shorter_than,
                ],
                cwd=trainer,
            )
            media_for_caption = scenes_dir
        else:
            media_for_caption = local_video

        # 3) Caption -> dataset.json
        if caption:
            cap_args: list[str] = [
                str(media_for_caption),
                "--output",
                str(dataset_path),
                "--captioner-type",
                captioner_type,
            ]
            if caption_use_8bit:
                cap_args.append("--use-8bit")
            typer.echo("Captioning...")
            run_py_script("caption_videos.py", cap_args, cwd=trainer)
    else:
        typer.echo(f"Using existing dataset manifest: {dataset_path}")

    if not dataset_path.is_file():
        typer.echo(
            f"No dataset at {dataset_path}. Enable --caption or pass --dataset-json.",
            err=True,
        )
        raise typer.Exit(code=1)

    # 4) Preprocess
    if preprocess:
        mp = Path(model_path).expanduser().resolve()
        if not mp.is_file():
            typer.echo(
                f"Missing LTX checkpoint for preprocessing: {mp}\n"
                "Download ltx-2.3-22b-dev.safetensors (or your variant) into models/ "
                "or pass --model-path to the real file.",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo("Preprocessing (latents + embeddings)...")
        pre_args: list[str] = [
            str(dataset_path),
            "--resolution-buckets",
            resolution_buckets,
            "--model-path",
            model_path,
            "--text-encoder-path",
            text_encoder_path,
            "--batch-size",
            str(preprocess_batch_size),
            "--device",
            device,
        ]
        if vae_tiling:
            pre_args.append("--vae-tiling")
        if with_audio:
            pre_args.append("--with-audio")
        if load_text_encoder_in_8bit:
            pre_args.append("--load-text-encoder-in-8bit")
        if lora_trigger:
            pre_args.extend(["--lora-trigger", lora_trigger])
        if decode:
            pre_args.append("--decode")
        run_py_script("process_dataset.py", pre_args, cwd=trainer)

    typer.echo("")
    typer.echo("Done.")
    typer.echo(f"  Dataset manifest: {dataset_path}")
    typer.echo(f"  Precomputed cache: {dataset_path.parent / '.precomputed'}")
    typer.echo("  Point your training config data path at the directory containing .precomputed (see configs/).")


if __name__ == "__main__":
    app()
