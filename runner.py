from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from itertools import cycle
from multiprocessing import get_context
from pathlib import Path

from tqdm.auto import tqdm


def find_wavs(input_dir: Path, recursive: bool) -> list[Path]:
  if not input_dir.exists() or not input_dir.is_dir():
    raise FileNotFoundError(f'Input directory missing: {input_dir}')
  files = input_dir.rglob('*.wav') if recursive else input_dir.glob('*.wav')
  wavs = sorted(p for p in files if p.is_file())
  # Strict: only wav files allowed
  others = [p for p in (input_dir.rglob('*') if recursive else input_dir.glob('*')) if p.is_file() and p.suffix.lower() != '.wav']
  if others:
    raise AssertionError('Only .wav files are allowed in the input directory.')
  return wavs


def existing_stems(out_dir: Path) -> set[str]:
  if not out_dir.exists():
    return set()
  if not out_dir.is_dir():
    raise NotADirectoryError(f'Output is not a directory: {out_dir}')
  return {p.stem for p in out_dir.rglob('*.json')}


def run_one(audio: Path, out_dir: str, model: str | None, device: str | None) -> None:
  """Run a single analysis via the allin1 CLI (silent)."""
  cmd = ['allin1', '--out-dir', out_dir, '--no-multiprocess']
  if model:
    cmd += ['--model', model]
  if device:
    cmd += ['--device', device]
  cmd.append(str(audio))

  with open(os.devnull, 'w') as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
    result = subprocess.run(cmd, stdout=devnull, stderr=devnull)
  if result.returncode != 0:
    raise RuntimeError(f'allin1 failed for {audio} (exit {result.returncode})')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  p = argparse.ArgumentParser('Batch runner for allin1')
  p.add_argument('input_dir', type=Path, help='Directory containing .wav files')
  p.add_argument('-o', '--output-dir', type=Path, default=Path('struct'), help='Where to store JSON (default: struct)')
  p.add_argument('-w', '--workers', type=int, default=2, help='Number of worker processes')
  p.add_argument('--recursive', action='store_true', help='Recurse into subdirectories')
  p.add_argument('--model', type=str, default=None, help='Optional model name')
  p.add_argument('--device', type=str, default=None, help='Optional device string (cpu / cuda / cuda:0)')
  p.add_argument('--dry-run', action='store_true', help='List tasks then exit')
  return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    import torch
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
  except Exception:
    gpu_count = 0

  wavs = find_wavs(args.input_dir, recursive=args.recursive)
  args.output_dir.mkdir(parents=True, exist_ok=True)

  done_stems = existing_stems(args.output_dir)
  todo = [w for w in wavs if w.stem not in done_stems]

  print(f'Found {len(wavs)} wav(s); skipping {len(wavs) - len(todo)} already processed; todo {len(todo)}')

  if args.dry_run:
    for w in todo:
      print(w)
    return 0
  if not todo:
    print('Nothing to do.')
    return 0
  if args.workers < 1:
    raise ValueError('--workers must be >= 1')

  device_pool = [f'cuda:{i % gpu_count}' for i in range(args.workers)] if gpu_count > 0 else ['cpu'] * args.workers
  if gpu_count == 0:
    pass  # silent; falls back to CPU

  # Spawn helps with torch/cuda
  ctx = get_context('spawn')
  with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
    devs = cycle(device_pool)
    futs = {ex.submit(run_one, w, str(args.output_dir), args.model, args.device or next(devs)): w for w in todo}
    failures: list[tuple[Path, Exception]] = []
    with tqdm(total=len(futs), desc='Processing', unit='file', dynamic_ncols=True) as pbar:
      for fut in as_completed(futs):
        w = futs[fut]
        try:
          fut.result()
        except Exception as exc:  # noqa: BLE001
          failures.append((w, exc))
        pbar.update(1)

  if failures:
    for w, exc in failures:
      print(f'[fail] {w}: {exc}')
    return 1
  return 0


if __name__ == '__main__':
  raise SystemExit(main())

