from __future__ import annotations

import argparse
import os
import platform
import shutil
import tarfile
import tempfile
from pathlib import Path

import requests
import zstandard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install Ollama into a user-writable directory without sudo.")
    parser.add_argument("--install-dir", required=True)
    parser.add_argument("--version", default=None, help="Optional Ollama version, for example 0.12.0.")
    return parser.parse_args()


def ollama_architecture() -> str:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "amd64"
    if machine in {"aarch64", "arm64"}:
        return "arm64"
    raise RuntimeError(f"Unsupported architecture: {machine}")


def validate_install_dir(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
        raise ValueError(f"Refusing unsafe install directory: {resolved}")


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with archive_path.open("rb") as compressed:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as archive:
                for member in archive:
                    member_path = (destination / member.name).resolve()
                    if destination != member_path and destination not in member_path.parents:
                        raise RuntimeError(f"Unsafe path in Ollama archive: {member.name}")
                    archive.extract(member, path=destination)


def download(url: str, destination: Path) -> None:
    downloaded = 0
    next_report = 256 * 1024 * 1024
    with requests.get(url, stream=True, timeout=(30, 600)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0"))
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    if total:
                        print(f"Downloaded {downloaded / (1024**3):.2f}/{total / (1024**3):.2f} GiB", flush=True)
                    else:
                        print(f"Downloaded {downloaded / (1024**3):.2f} GiB", flush=True)
                    next_report += 256 * 1024 * 1024


def main() -> None:
    args = parse_args()
    install_dir = Path(args.install_dir).expanduser().resolve()
    validate_install_dir(install_dir)
    install_dir.parent.mkdir(parents=True, exist_ok=True)

    binary = install_dir / "bin" / "ollama"
    if binary.is_file() and os.access(binary, os.X_OK):
        print(f"Ollama is already installed at {binary}")
        return

    architecture = ollama_architecture()
    url = f"https://ollama.com/download/ollama-linux-{architecture}.tar.zst"
    if args.version:
        url = f"{url}?version={args.version}"

    staging_dir = Path(tempfile.mkdtemp(prefix="ollama-install-", dir=install_dir.parent)).resolve()
    archive_path = staging_dir / "ollama.tar.zst"
    extracted_dir = staging_dir / "root"
    extracted_dir.mkdir()

    try:
        print(f"Downloading {url}", flush=True)
        download(url, archive_path)
        print(f"Extracting Ollama to {install_dir}", flush=True)
        safe_extract(archive_path, extracted_dir)

        extracted_binary = extracted_dir / "bin" / "ollama"
        if not extracted_binary.is_file():
            raise RuntimeError("Downloaded archive does not contain bin/ollama")

        if install_dir.exists():
            shutil.rmtree(install_dir)
        extracted_dir.replace(install_dir)
        binary.chmod(binary.stat().st_mode | 0o111)
        print(f"Installed Ollama without sudo: {binary}", flush=True)
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)


if __name__ == "__main__":
    main()
