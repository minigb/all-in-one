#!/usr/bin/env python3
"""
MP3 to WAV Batch Converter (FFmpeg version)
Converts all MP3 files in a directory to WAV format with duplication checking.
This version uses subprocess to call ffmpeg directly, avoiding the pydub dependency.
"""

from tqdm import tqdm
import subprocess
import argparse
from pathlib import Path
from typing import List, Union


def check_ffmpeg_installed():
    """Check if ffmpeg is installed and available in PATH."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


# Default output directory if none provided
DEFAULT_OUTPUT_DIR = Path.cwd() / "wav_files"


def convert_mp3_to_wav_ffmpeg(
    mp3_path: Union[str, Path],
    output_dir: Union[str, Path] = None,
    skip_existing: bool = True,
    verbose: bool = True,
    sample_rate: int = None,
    channels: int = None
) -> str:
    """
    Convert a single MP3 file to WAV format using ffmpeg.
    
    Args:
        mp3_path: Path to the MP3 file
        output_dir: Directory to save the WAV file. If None, saves in DEFAULT_OUTPUT_DIR.
        skip_existing: If True, skip conversion if WAV file already exists
        verbose: If True, print conversion status
        sample_rate: Output sample rate in Hz (e.g., 44100, 48000). If None, preserves original.
        channels: Number of output channels (1=mono, 2=stereo). If None, preserves original.
    
    Returns:
        Path to the converted WAV file
    """
    mp3_path = Path(mp3_path)
    
    if not mp3_path.exists():
        raise FileNotFoundError(f"MP3 file not found: {mp3_path}")
    
    # Determine output path
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / mp3_path.with_suffix('.wav').name
    
    # Check if WAV file already exists
    if wav_path.exists() and skip_existing:
        if verbose:
            print(f"⏭️  Skipping (already exists): {mp3_path.name} -> {wav_path.name}")
        return str(wav_path)
    
    try:
        # Build ffmpeg command
        cmd = ['ffmpeg', '-i', str(mp3_path)]
        
        # Add sample rate if specified
        if sample_rate is not None:
            cmd.extend(['-ar', str(sample_rate)])
        
        # Add channels if specified
        if channels is not None:
            cmd.extend(['-ac', str(channels)])
        
        # Add output options
        cmd.extend([
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-y',  # Overwrite output file if it exists
            str(wav_path)
        ])
        
        if verbose:
            print(f"🔄 Converting: {mp3_path.name} -> {wav_path.name}")
        
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr}")
        
        if verbose:
            print(f"✅ Converted successfully: {wav_path.name}")
        
        return str(wav_path)
    
    except Exception as e:
        print(f"❌ Error converting {mp3_path.name}: {str(e)}")
        raise


def batch_convert_directory(
    directory: Union[str, Path],
    output_dir: Union[str, Path] = None,
    skip_existing: bool = True,
    recursive: bool = False,
    verbose: bool = True,
    sample_rate: int = None,
    channels: int = None
) -> List[str]:
    """
    Convert all MP3 files in a directory to WAV format.
    
    Args:
        directory: Path to the directory containing MP3 files
        output_dir: Directory to save the WAV files. If None, uses DEFAULT_OUTPUT_DIR.
        skip_existing: If True, skip conversion if WAV file already exists
        recursive: If True, search for MP3 files recursively in subdirectories
        verbose: If True, print conversion status
        sample_rate: Output sample rate in Hz. If None, preserves original.
        channels: Number of output channels. If None, preserves original.
    
    Returns:
        List of paths to converted WAV files
    """
    directory = Path(directory)
    
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")
    
    # Find all MP3 files
    if recursive:
        mp3_files = list(directory.rglob('*.mp3')) + list(directory.rglob('*.MP3'))
    else:
        mp3_files = list(directory.glob('*.mp3')) + list(directory.glob('*.MP3'))
    
    # Remove duplicates (in case both .mp3 and .MP3 exist)
    mp3_files = list(set(mp3_files))
    mp3_files.sort()
    
    if not mp3_files:
        print(f"⚠️  No MP3 files found in {directory}")
        return []
    
    print(f"\n📁 Found {len(mp3_files)} MP3 file(s) in {directory}")
    print("=" * 60)
    
    converted_files = []
    error_count = 0
    
    for mp3_file in tqdm(mp3_files):
        try:
            # Determine output directory for recursive mode
            if recursive and output_dir is not None:
                # Preserve subdirectory structure
                relative_path = mp3_file.relative_to(directory)
                output_subdir = Path(output_dir) / relative_path.parent
                wav_path = convert_mp3_to_wav_ffmpeg(
                    mp3_file, 
                    output_dir=output_subdir, 
                    skip_existing=skip_existing,
                    verbose=verbose,
                    sample_rate=sample_rate,
                    channels=channels
                )
            else:
                wav_path = convert_mp3_to_wav_ffmpeg(
                    mp3_file, 
                    output_dir=output_dir, 
                    skip_existing=skip_existing,
                    verbose=verbose,
                    sample_rate=sample_rate,
                    channels=channels
                )
            
            converted_files.append(wav_path)
            
        except Exception as e:
            error_count += 1
            if verbose:
                print(f"❌ Failed to convert {mp3_file.name}: {str(e)}")
    
    
    # Summary
    print("=" * 60)
    print(f"\n📊 Conversion Summary:")
    print(f"   Total MP3 files found: {len(mp3_files)}")
    print(f"   Total processed: {len(converted_files)}")
    print(f"   Errors: {error_count}")
    
    return converted_files


def main():
    """Main function for CLI usage."""
    # Check if ffmpeg is installed
    if not check_ffmpeg_installed():
        print("❌ Error: ffmpeg is not installed or not in PATH")
        print("\nPlease install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        return
    
    parser = argparse.ArgumentParser(
        description="Convert MP3 files to WAV format with duplication checking (using ffmpeg)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all MP3 files in current directory
  python mp3_to_wav_converter_ffmpeg.py .
  
  # Convert MP3 files from a specific directory
  python mp3_to_wav_converter_ffmpeg.py /path/to/mp3/folder
  
  # Convert and save to a different output directory
  python mp3_to_wav_converter_ffmpeg.py ./mp3s --output ./wavs
  
  # Convert recursively through subdirectories
  python mp3_to_wav_converter_ffmpeg.py ./music --recursive
  
  # Force reconvert even if WAV files exist
  python mp3_to_wav_converter_ffmpeg.py ./mp3s --no-skip-existing
  
  # Convert a single MP3 file
  python mp3_to_wav_converter_ffmpeg.py ./song.mp3
  
  # Convert with specific sample rate and mono output
  python mp3_to_wav_converter_ffmpeg.py ./mp3s --sample-rate 44100 --channels 1
        """
    )
    
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to MP3 file or directory containing MP3 files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for WAV files (default: same as input)'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Search for MP3 files recursively in subdirectories'
    )
    
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Reconvert even if WAV file already exists'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=None,
        help='Output sample rate in Hz (e.g., 44100, 48000). Default: preserve original'
    )
    
    parser.add_argument(
        '--channels',
        type=int,
        choices=[1, 2],
        default=None,
        help='Number of output channels: 1=mono, 2=stereo. Default: preserve original'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output) if args.output is not None else DEFAULT_OUTPUT_DIR
    skip_existing = not args.no_skip_existing
    verbose = not args.quiet
    
    # Check if input is a file or directory
    if input_path.is_file():
        # Convert single file
        if input_path.suffix.lower() != '.mp3':
            print(f"❌ Error: Input file is not an MP3 file: {input_path}")
            return
        
        try:
            convert_mp3_to_wav_ffmpeg(
                input_path,
                output_dir=output_dir,
                skip_existing=skip_existing,
                verbose=verbose,
                sample_rate=args.sample_rate,
                channels=args.channels
            )
        except Exception as e:
            print(f"❌ Conversion failed: {str(e)}")
            return
    
    elif input_path.is_dir():
        # Convert directory
        try:
            batch_convert_directory(
                input_path,
                output_dir=output_dir,
                skip_existing=skip_existing,
                recursive=args.recursive,
                verbose=verbose,
                sample_rate=args.sample_rate,
                channels=args.channels
            )
        except Exception as e:
            print(f"❌ Batch conversion failed: {str(e)}")
            return
    
    else:
        print(f"❌ Error: Path does not exist: {input_path}")
        return


if __name__ == "__main__":
    main()

