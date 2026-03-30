#!/usr/bin/env python3

"""
Copy every video referenced in a Labelbox-style golden JSON to a destination GCS prefix.

Reads a manifest (JSON array of `{ "data": { "video": "gs://...", "meta": {...} } }`),
runs `gsutil cp` for each source object to a path under `--dest`, and optionally uploads an
updated manifest whose `data.video` fields point at the new locations.

Example:

    uv run python scripts/prep_data_1_sync_golden_manifest_to_gcs.py \\
        --manifest gs://video_gen_dataset/dataset/labelbox/skye_golden.json \\
        --dest gs://video_gen_dataset/TinyStories/data_sets/golden_skye

Requires `gsutil` on PATH with credentials that can read the manifest and source objects
and write to the destination prefix.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help="Sync golden manifest videos to a GCS prefix.")


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """Return (bucket, object_name) for gs://bucket/object."""
    if not uri.startswith("gs://"):
        msg = f"Expected gs:// URI, got: {uri!r}"
        raise ValueError(msg)
    rest = uri[5:]
    slash = rest.find("/")
    if slash == -1:
        return rest, ""
    return rest[:slash], rest[slash + 1 :]


def normalize_dest_prefix(dest: str) -> str:
    """Accept 'bucket/path' or 'gs://bucket/path'; return gs:// URI without trailing slash."""
    d = dest.strip()
    if not d.startswith("gs://"):
        d = f"gs://{d}"
    return d.rstrip("/")


def source_to_dest_object(source_uri: str, dest_prefix: str, strip_prefix: str) -> str:
    """
    Map source object URI to full destination gs:// URI.

    By default strips `dataset/` from the blob path so copies land under
    `.../golden_skye/paw_patrol/...` instead of `.../golden_skye/dataset/paw_patrol/...`.
    """
    _, blob = parse_gs_uri(source_uri)
    if strip_prefix and blob.startswith(strip_prefix):
        rel = blob[len(strip_prefix) :].lstrip("/")
    else:
        rel = blob
    if not rel:
        msg = f"Empty relative path after strip for {source_uri!r}"
        raise ValueError(msg)
    return f"{dest_prefix}/{rel}"


def gsutil_cat(uri: str) -> bytes:
    return subprocess.check_output(["gsutil", "cat", uri], stderr=subprocess.PIPE)


def gsutil_cp(src: str, dst: str) -> tuple[str, str | None]:
    """Copy src to dst. Returns (dst, error_message_or_none)."""
    try:
        subprocess.run(
            ["gsutil", "cp", src, dst],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        err = e.stderr or e.stdout or str(e)
        return dst, err
    return dst, None


def object_exists(uri: str) -> bool:
    try:
        subprocess.run(
            ["gsutil", "ls", uri],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


@app.command()
def main(
    manifest: str = typer.Option(
        "gs://video_gen_dataset/dataset/labelbox/skye_golden.json",
        "--manifest",
        "-m",
        help="gs:// URI of the Labelbox-style JSON manifest",
    ),
    dest: str = typer.Option(
        "gs://video_gen_dataset/TinyStories/data_sets/golden_skye",
        "--dest",
        "-d",
        help="Destination GCS prefix (bucket/path, with or without gs://)",
    ),
    strip_prefix: str = typer.Option(
        "dataset/",
        "--strip-prefix",
        help="Strip this blob prefix when building destination object paths (after bucket)",
    ),
    workers: int = typer.Option(
        16,
        "--workers",
        "-j",
        help="Parallel gsutil cp jobs",
        min=1,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print planned copies only; do not run gsutil cp",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip copy if destination object already exists",
    ),
    write_manifest: bool = typer.Option(
        True,
        "--write-manifest/--no-write-manifest",
        help="Upload skye_golden.json under --dest with updated data.video URIs",
    ),
    manifest_name: str = typer.Option(
        "skye_golden.json",
        "--manifest-name",
        help="Object name for the uploaded manifest when --write-manifest is set",
    ),
) -> None:
    """Copy manifest videos to dest prefix and optionally upload an updated JSON manifest."""
    dest_prefix = normalize_dest_prefix(dest)

    raw = gsutil_cat(manifest)
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, list):
        typer.echo("Manifest root must be a JSON array.", err=True)
        raise typer.Exit(code=1)

    tasks: list[tuple[str, str]] = []
    for i, row in enumerate(data):
        try:
            src = row["data"]["video"]
        except (KeyError, TypeError) as e:
            typer.echo(f"Record {i}: missing data.video: {e}", err=True)
            raise typer.Exit(code=1) from e
        if not isinstance(src, str) or not src.startswith("gs://"):
            typer.echo(f"Record {i}: invalid data.video: {src!r}", err=True)
            raise typer.Exit(code=1)
        dst_uri = source_to_dest_object(src, dest_prefix, strip_prefix)
        tasks.append((src, dst_uri))

    typer.echo(f"Manifest: {manifest}")
    typer.echo(f"Destination prefix: {dest_prefix}")
    typer.echo(f"Videos to copy: {len(tasks)}")

    if dry_run:
        for src, dst_uri in tasks[:20]:
            typer.echo(f"  {src} -> {dst_uri}")
        if len(tasks) > 20:
            typer.echo(f"  ... and {len(tasks) - 20} more")
        raise typer.Exit(code=0)

    errors: list[tuple[str, str]] = []
    done = 0

    def run_one(pair: tuple[str, str]) -> tuple[str, str | None]:
        src, dst_uri = pair
        if skip_existing and object_exists(dst_uri):
            return dst_uri, None
        return gsutil_cp(src, dst_uri)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            dst_uri, err = fut.result()
            done += 1
            if err:
                src = futures[fut][0]
                errors.append((src, err))
            if done % 50 == 0 or done == len(tasks):
                typer.echo(f"Progress: {done}/{len(tasks)}")

    if errors:
        typer.echo(f"Failed copies: {len(errors)}", err=True)
        for src, err in errors[:30]:
            typer.echo(f"  {src}\n    {err[:500]}", err=True)
        if len(errors) > 30:
            typer.echo(f"  ... and {len(errors) - 30} more errors", err=True)
        raise typer.Exit(code=1)

    typer.echo("All video copies finished.")

    if write_manifest:
        for i, row in enumerate(data):
            row["data"]["video"] = tasks[i][1]

        out_json = json.dumps(data, indent=2)
        out_uri = f"{dest_prefix}/{manifest_name}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp.write(out_json)
            tmp_path = tmp.name
        try:
            subprocess.run(
                ["gsutil", "cp", tmp_path, out_uri],
                check=True,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        typer.echo(f"Uploaded updated manifest: {out_uri}")


if __name__ == "__main__":
    app()
