"""
This script provides functionality for transcribing audio files into text
using the Whisper model.

It includes the following core features:

1. **Audio File Handling**:
   - Traverse directories recursively to find audio files (e.g., .mp3, .mp4).
   - Filter audio files that have not been processed or failed previously.

2. **Transcription**:
   - Use OpenAI's Whisper model to transcribe audio files into text.
   - Optionally include timestamps in the transcriptions.
   - Handle transcription failures and continue processing remaining files.

3. **State Management**:
   - Track processed and failed files between runs to avoid redundant work.
   - Save and load the processing state from a JSON file.

4. **File Structure Management**:
   - Optionally store transcriptions in a separate directory or subfolder.
   - Handle the creation of necessary directories for transcription files.

5. **Command-Line Interface (CLI)**:
   - Provide a command-line interface for users to specify input paths,
     transcription settings, and model options.
   - Support verbose mode for displaying progress and including timestamps
     in the transcriptions.

The module is designed to be flexible and efficient, enabling batch
transcription of audio files with the ability to resume interrupted processes.

Usage:
    To use the module, run the script with command-line arguments specifying
    paths to the audio files or directories, transcription settings, and the
    Whisper model.

    Example:
        python transcribe.py -m base --verbose --ignore-existing /path/to/audio/files

Requirements:
    - `Whisper`: The OpenAI speech recognition model library. Install with
      `pip install -U openai-whisper`
    - `structlog`: Structured logging library. Install with
      `pip install structlog`
"""

import argparse
from dataclasses import dataclass
import itertools
import json
import logging
from pathlib import Path
import re
import typing
import warnings

import structlog
from structlog.contextvars import bound_contextvars
import whisper

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

LOGGER = structlog.get_logger()

# Suppress FP16 warning
warnings.filterwarnings(
    "ignore", message="FP16 is not supported on CPU; using FP32 instead"
)

CURRENT_DIR = Path(__file__).parent

# _______________ Saving and Loading Progress ______________

# ------------------- Paths to/from JSON -------------------


def state_paths(
    state: dict[str, list[str]],
    state_key: str,
) -> set[Path]:
    """
    Retrieve a set of `Path` objects from the given state dictionary for a
    specific key.

    This function looks up the list of string paths associated with
    `state_key` in the `state` dictionary, converts each string into a
    `Path` object, and returns them as a set.

    Args:
        state (dict[str, list[str]]): A dictionary representing the current
            state, where each key maps to a list of file path strings.
        state_key (str): The key in the state dictionary to look up.

    Returns:
        set[Path]: A set of `Path` objects corresponding to the file paths
        listed under `state_key`. Returns an empty set if the key is not found.
    """
    state_paths = state.get(state_key, [])
    paths = (Path(item) for item in state_paths)
    return set(paths)


# -------------------- Saving / Loading --------------------


@dataclass
class State:
    """
    Represents the state of processed and failed file paths.

    Attributes:
        processed (set[Path]): Set of file paths that have been successfully
            processed.
        failed (set[Path]): Set of file paths that failed during processing.
    """

    processed: set[Path]
    failed: set[Path]


def new_state() -> State:
    """
    Create a new, empty state.

    Returns:
        State: A new State instance with empty processed and failed sets.
    """
    return State(processed=set(), failed=set())


def json_to_state(state_json: dict[str, list[str]]) -> State:
    """
    Convert a dictionary to a State object.

    Args:
        state_json (dict[str, list[str]]): A dictionary with "processed" and/or
            "failed" keys mapping to lists of file path strings.

    Returns:
        State: A State instance with the corresponding paths as `Path` objects.
    """
    processed = state_paths(state=state_json, state_key="processed")
    failed = state_paths(state=state_json, state_key="failed")
    return State(processed=processed, failed=failed)


def state_to_json(state: State) -> dict[str, list[str]]:
    """
    Convert a State object to a JSON-serializable dictionary.

    Args:
        state (State): The state to convert.

    Returns:
        dict[str, list[str]]: A dictionary with "processed" and "failed" keys
        mapping to lists of stringified file paths.
    """
    processed = [str(path) for path in state.processed]
    failed = [str(path) for path in state.failed]
    return {"processed": processed, "failed": failed}


def update_state(
    state: State,
    processed: set[Path],
    failed: set[Path],
) -> State:
    """
    Update the given state with additional processed and failed paths.

    Args:
        state (State): The original state.
        processed (set[Path]): New successfully processed file paths.
        failed (set[Path]): New failed file paths.

    Returns:
        State: A new State with the updated processed and failed sets.
    """
    new_processed = state.processed.union(processed)
    new_failed = state.failed.union(failed)
    state = State(
        processed=new_processed,
        failed=new_failed,
    )
    LOGGER.info(
        "Updated state paths",
        processed_count=len(new_processed),
        failed_count=len(new_failed),
    )
    return state


def load_state(state_path: Path) -> State:
    """
    Load state from a JSON file.

    Args:
        state_path (Path): The path to the JSON file storing the state.

    Returns:
        State: The loaded state, or a new empty state if the file does not
            exist.
    """
    if state_path.exists():
        with state_path.open(mode="r", encoding="utf-8") as fp:
            state_json = json.load(fp)
        return json_to_state(state_json=state_json)
    return new_state()


def save_state(
    state_path: Path,
    state: State,
):
    """
    Save the current state to a JSON file.

    Args:
        state_path (Path): The path where the state should be saved.
        state (State): The state to serialize and write.
    """
    with bound_contextvars(state_path=state_path):
        LOGGER.info("Saving state")
        with state_path.open(mode="w", encoding="utf-8") as fp:
            state_json = state_to_json(state)
            json.dump(state_json, fp, indent=4)
        LOGGER.info("Saved state")


def create_transcripts_structure(
    folder_path: Path,
):
    """
    Create a new transcripts folder structure or reuse the input folder.

    Args:
        folder_path (Path): The base path for creating the transcripts folder.
        same_folder (bool): If True, reuse `folder_path` instead of creating
            a new folder.

    Returns:
        Path: The path to the transcripts folder.
    """
    transcripts_dir = folder_path.parent / f"{folder_path.name}_transcripts"
    if not transcripts_dir.exists():
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Created transcripts folder: {transcripts_dir}")
    return transcripts_dir


# ------------------ Regular State Saving ------------------


def countdown(count: int):
    """
    Create an infinite countdown cycle starting from `count - 1` down to 0.

    This function returns an infinite iterator that repeatedly counts down
    from `count - 1` to 0, then restarts from `count - 1` again. This means
    the cycle is `count` periods long.

    Args:
        count (int): The number to start the countdown from. The countdown
            begins at `count - 1` and includes 0.

    Returns:
        itertools.cycle: An infinite iterator cycling through the countdown
            sequence.

    Example:
        >>> counter = countdown(5)
        >>> [next(counter) for _ in range(10)]
        [4, 3, 2, 1, 0, 4, 3, 2, 1, 0]
    """
    return itertools.cycle(range(count - 1, -1, -1))


# _______________ File / Directory Operations ______________

# ----------------- Retrieving Input Files -----------------


def descendants(ancestor: Path) -> typing.Generator[Path, None, None]:
    """
    Recursively yield all descendant files and directories of a given path.

    If the ancestor is a file, it yields the file itself.
    If it's a directory, it yields the directory and all its contents
    recursively.

    Args:
        ancestor (Path): The root path from which to yield descendants.

    Yields:
        Path: Each descendant path, including the ancestor itself if it's a
            file or directory.
    """
    if ancestor.is_dir():
        yield ancestor
        for child in ancestor.iterdir():
            yield from descendants(ancestor=child)
    else:
        yield ancestor


FILE_EXTENSIONS_AUDIO = (".mp4", ".mp3", ".mov", ".m4a", ".ogg")


def filter_audio_files(
    files: typing.Iterable[Path],
) -> typing.Generator[Path, None, None]:
    """
    Filter an iterable of paths to yield only audio files.

    Audio files are identified by common extensions such as .mp4, .mp3, etc.

    Args:
        files (Iterable[Path]): An iterable of file paths to filter.

    Yields:
        Path: Each path that has an audio file extension.
    """
    for file in files:
        if file.suffix.lower() in FILE_EXTENSIONS_AUDIO:
            yield file


@dataclass
class MediaFile:
    """
    Represents an audio input and its corresponding transcription output.

    Attributes:
        audio_input (Path): The original audio file.
        transcription_output (Path): The output path for the transcription
            file.
    """

    audio_input: Path
    transcription_output: Path


# ............... Check if MediaFile in State ..............


def filter_processed_media_files(
    state: State,
    media_files: typing.Iterable[MediaFile],
) -> typing.Generator[MediaFile, None, None]:
    """
    Yield only media files that have not been processed or failed before.

    Logs the status of each file:
    - Skipping processed
    - Skipping failed
    - Processing (to be yielded)

    Args:
        state (State): The current processing state with processed and failed
            paths.
        media_files (Iterable[MediaFile]): A list of media files to check.

    Yields:
        MediaFile: Each file that hasn't been previously processed or failed.
    """
    for media_file in media_files:
        audio_input = media_file.audio_input
        with bound_contextvars(audio_input=audio_input):
            if audio_input in state.processed:
                LOGGER.info(f"Skipping processed")
                continue
            if audio_input in state.failed:
                LOGGER.info(f"Skipping failed")
            LOGGER.info("Processing")
            yield media_file


def media_file_in_transcript_dir(
    audio_file: Path,
    transcripts_dir: Path,
) -> MediaFile:
    """
    Generate a MediaFile where the transcript is stored in a flat transcripts
    directory.

    The output filename is sanitized and derived from the full path of the
    audio file.

    Args:
        audio_file (Path): The original audio file path.
        transcripts_dir (Path): The directory to store transcript files.

    Returns:
        MediaFile: A MediaFile with the audio input and corresponding
        transcript output path.
    """
    filename = "_".join(audio_file.parts)
    filename = re.sub(
        r"[^\w_. -]+",
        "",
        filename,
    )
    transcription_output = (transcripts_dir / filename).with_suffix(".txt")
    return MediaFile(
        audio_input=audio_file,
        transcription_output=transcription_output,
    )


def media_file_in_subfolder(audio_file: Path) -> MediaFile:
    """
    Generate a MediaFile where the transcript is stored next to the audio file.

    Args:
        audio_file (Path): The original audio file path.

    Returns:
        MediaFile: A MediaFile with the transcript output set to the same
            folder.
    """
    transcription_output = audio_file.with_suffix(".txt")
    return MediaFile(
        audio_input=audio_file,
        transcription_output=transcription_output,
    )


def media_files_to_origin_dir(
    folder_path: Path,
    ignore_existing: bool,
) -> typing.Generator[MediaFile, None, None]:
    """
    Discover all audio files and generate MediaFile objects that will be
    stored in the same directory as the original audio file.

    Args:
        folder_path (Path): Path to the folder containing audio files.
        ignore_existing (bool): Whether to skip files with existing
            transcripts.

    Yields:
        MediaFile: Each MediaFile corresponding to a discovered audio file.
    """
    files = descendants(ancestor=folder_path)
    audio_files = filter_audio_files(files=files)

    for audio_file in audio_files:
        media_file = media_file_in_subfolder(audio_file=audio_file)

        transcription_output = media_file.transcription_output
        if ignore_existing and transcription_output.exists():
            LOGGER.info(
                f"Skipping audio file, transcript already exists",
                audio_input=audio_file,
                transcription_output=transcription_output,
            )
            continue

        yield media_file


def media_files_to_transcripts_dir(
    folder_path: Path,
    transcripts_dir: Path,
    ignore_existing: bool,
) -> typing.Generator[MediaFile, None, None]:
    """
    Discover all audio files and generate MediaFile objects.

    Args:
        folder_path (Path): Path to the folder containing audio files.
        transcripts_path (Path): Directory to store transcript files
            (used if store_in_subfolders is False).
        ignore_existing (bool): Whether to skip files with existing
            transcripts.
        store_in_subfolders (bool): Whether to store transcripts in
            the same folders as audio files.If False, all transcripts are
            stored in `transcripts_path`.

    Yields:
        MediaFile: Each MediaFile corresponding to a discovered audio file.
    """
    files = descendants(ancestor=folder_path)
    audio_files = filter_audio_files(files=files)

    for audio_file in audio_files:
        media_file = media_file_in_transcript_dir(
            audio_file=audio_file, transcripts_dir=transcripts_dir
        )

        transcription_output = media_file.transcription_output
        if ignore_existing and transcription_output.exists():
            LOGGER.info(
                f"Skipping audio file, transcript already exists",
                audio_input=audio_file,
                transcription_output=transcription_output,
            )
            continue

        yield media_file


# ---------------------- Transcribing ----------------------

# ................ Transcription Data Types ................


@dataclass
class TranscriptionOk:
    """
    Represents a successful transcription result.

    Attributes:
        transcription (str): The transcribed text.
    """

    transcription: str


@dataclass
class TranscriptionErr:
    """
    Represents a failed transcription result.

    Attributes:
        reason (Exception): The exception or error that occurred.
    """

    reason: Exception


TranscriptionResult = TranscriptionOk | TranscriptionErr

# ................ Transcription Formatting ................


def format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder_seconds = seconds % 60
    return f"{minutes:02d}:{remainder_seconds:06.3f}"


# .............. Transcribing Audio From File ..............


def transcribe_audio(
    model: whisper.Whisper,
    media_file: MediaFile,
    verbose: typing.Optional[bool],
) -> TranscriptionResult:
    """
    Transcribes an audio file using the given Whisper model.

    Args:
        model (whisper.Whisper): The Whisper model instance used for
            transcription.
        media_file (MediaFile): The media file containing the audio input and
            target output path.
        verbose (Optional[bool]): Whether to show detailed output during
            transcription.

    Returns:
        TranscriptionResult: Either a TranscriptionOk with the transcription
            string, or a TranscriptionErr with the error encountered.
    """
    LOGGER.info(f"Transcribing media file")
    try:
        result = model.transcribe(
            audio=str(media_file.audio_input),
            language="en",
            verbose=verbose,
        )
    except FileNotFoundError as err:
        LOGGER.info("Failed to transcribe media file", err=err)
        return TranscriptionErr(reason=err)

    text = result["text"]
    transcription = text.replace(". ", ".\n\n")
    LOGGER.info("Successfully transcribed media file")
    return TranscriptionOk(transcription=transcription)


def format_segment(segment: dict[str, str]) -> str:
    """Formats a transcription timestamped segment into a line.

    Args:
        segment (dict[str, str]): The segment from the transcription.

    Returns:
        str: The formatted line
    """
    start = segment["start"]
    end = segment["end"]
    LOGGER.debug("Processing segment", start=start, end=end)
    text = segment["text"].strip()
    start_time = format_timestamp(seconds=start)
    end_time = format_timestamp(seconds=end)
    return f"[{start_time} --> {end_time}]  {text}"


def transcribe_audio_timestamped(
    model: whisper.Whisper,
    media_file: MediaFile,
    verbose: typing.Optional[bool],
) -> TranscriptionResult:
    """
    Transcribes an audio file using the given Whisper model.

    Args:
        model (whisper.Whisper): The Whisper model instance used for
            transcription.
        media_file (MediaFile): The media file containing the audio input and
            target output path.
        verbose (Optional[bool]): Whether to show detailed output during
            transcription.

    Returns:
        TranscriptionResult: Either a TranscriptionOk with the transcription
            string, or a TranscriptionErr with the error encountered.
    """
    LOGGER.info(f"Transcribing media file")
    try:
        result = model.transcribe(
            audio=str(media_file.audio_input),
            language="en",
            verbose=verbose,
        )
    except FileNotFoundError as err:
        LOGGER.info("Failed to transcribe media file", err=err)
        return TranscriptionErr(reason=err)

    processed_segments = (
        format_segment(segment=segment) for segment in result["segments"]
    )
    transcription = "\n".join(processed_segments) + "\n"
    LOGGER.info("Successfully transcribed media file")
    return TranscriptionOk(transcription=transcription)


# .............. Saving Transcription To File ..............


def write_transcription(transcription: str, media_file: MediaFile):
    """
    Writes the transcription text to the corresponding output file.

    Args:
        transcription (str): The transcription text to write.
        media_file (MediaFile): The media file object containing the output
            file path.

    Side Effects:
        Creates the output directory if it doesn't exist and writes the
        transcription to a .txt file.
    """
    transcription_output = media_file.transcription_output
    with bound_contextvars(transcription_output=transcription_output):
        LOGGER.info("Writing transcription to file")
        transcription_output.parent.mkdir(parents=True, exist_ok=True)
        with transcription_output.open(mode="w", encoding="utf-8") as fp:
            fp.write(transcription)
        LOGGER.info("Wrote transcription to file")


# ______________________ Command Line ______________________

# ----------------- Command Line Arguments -----------------


def setup_argparser() -> argparse.ArgumentParser:
    """
    Sets up and returns the argument parser for the transcription CLI
    application.

    Returns:
        argparse.ArgumentParser: The argument parser configured with all
        supported CLI options:

            -m / --model: Whisper model to use for transcription.
            -t / --transcript-dir: Directory to save transcript files.
            --include-timestamps: Whether to include timestamps in the
                transcription.
            -v / --verbose: Level of verbosity for the transcription process.
            --ignore-existing: Whether to skip transcription for files that
                already have transcript files.
            paths: List of input file or directory paths to transcribe.
    """
    parser = argparse.ArgumentParser(
        prog="Transcribe",
        description="Transcribes a list of files into TXT files for LLM use.",
    )
    parser.add_argument(
        "-m",
        "--model",
        nargs="?",
        default="base",
        help="The Whisper model to use for transcription (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--transcript-dir",
        nargs="?",
        const=CURRENT_DIR / "transcripts",
        default=None,
        help="""The directory to save transcripts under. If this flag is not
            set, transcript files will be saved next to the audio file inputs.
            If this flag is provided without an argument, defaults to a folder
            "transcripts" under the same directory as this script.
        """,
        type=Path,
    )
    parser.add_argument(
        "--include-timestamps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include timestamps in the transcribed text files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="""Display the progress of the Whisper transcription to the
        console. If this flag is not set, no progress bar will be displayed.
        If this flag is set with --verbose or -v, a small display bar will be
        shown. If this flag is set with -vv, the audio transcripts will 
        be logged to the console.
        """,
    )
    parser.add_argument(
        "--ignore-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore already existing transcription files",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="""The list of paths to directories (to transcribe all descendant
        files) and files (to transcribe)""",
    )
    return parser


# ---------------- Command Line Application ----------------

STATE_PATH = CURRENT_DIR / "transcribe_for_llm_state.json"


def main():
    """
    Main entry point for the transcription CLI application.

    This function:
    1. Parses CLI arguments.
    2. Loads the Whisper model and previous processing state.
    3. Gathers audio files from given paths.
    4. Filters out already processed or failed files.
    5. Transcribes the unprocessed files using the Whisper model.
    6. Writes successful transcriptions to disk.
    7. Updates and saves the processing state regularly and at the end.

    Side Effects:
        - Loads a machine learning model.
        - Reads/writes to disk (state file and transcription files).
        - LOGGER.infos progress and errors to stdout.
    """
    # Get CLI arguments
    args = setup_argparser().parse_args()
    transcript_dir = args.transcript_dir
    include_timestamps = args.include_timestamps
    store_in_subfolders = transcript_dir == None
    ignore_existing = args.ignore_existing
    if args.verbose == 0:
        verbose = None
    elif args.verbose == 1:
        verbose = False
    elif args.verbose >= 2:
        verbose = True
    paths = [Path(path) for path in args.paths]

    LOGGER.debug(
        "Starting main",
        transcript_dir=transcript_dir,
        include_timestamps=include_timestamps,
        store_in_subfolders=store_in_subfolders,
        ignore_existing=ignore_existing,
        verbose=verbose,
    )

    # Load state and Whisper model
    model = whisper.load_model(args.model)
    state = load_state(state_path=STATE_PATH)

    # Get all media files

    if store_in_subfolders:
        media_files = itertools.chain.from_iterable(
            [
                media_files_to_origin_dir(
                    folder_path=path,
                    ignore_existing=ignore_existing,
                )
                for path in paths
                if path.exists()
            ]
        )
    else:
        media_files = media_files = itertools.chain.from_iterable(
            [
                media_files_to_transcripts_dir(
                    folder_path=path,
                    transcripts_dir=transcript_dir,
                    ignore_existing=ignore_existing,
                )
                for path in paths
                if path.exists()
            ]
        )

    transcribe = (
        transcribe_audio_timestamped if include_timestamps else transcribe_audio
    )

    unprocessed = filter_processed_media_files(
        state=state,
        media_files=media_files,
    )

    current_processed = set()
    current_failed = set()
    # Transcribe media files
    for count, media_file in zip(countdown(count=5), unprocessed):
        with bound_contextvars(audio_input=media_file.audio_input):
            transcript = transcribe(
                model=model,
                media_file=media_file,
                verbose=verbose,
            )
            if isinstance(transcript, TranscriptionOk):
                current_processed.add(media_file.audio_input)
                write_transcription(
                    transcription=transcript.transcription,
                    media_file=media_file,
                )
            elif isinstance(transcript, TranscriptionErr):
                current_failed.add(media_file.audio_input)
        if count == 0:
            state = update_state(
                state=state,
                processed=current_processed,
                failed=current_failed,
            )
            save_state(state_path=STATE_PATH, state=state)
    LOGGER.info(
        "Processed files count",
        processed_count=len(current_processed),
    )
    LOGGER.info(
        "Failed files count",
        failed_count=len(current_failed),
    )
    # Save state
    state = update_state(
        state=state,
        processed=current_processed,
        failed=current_failed,
    )
    save_state(state_path=STATE_PATH, state=state)
    LOGGER.info(
        "Total processed files count",
        processed_count=len(state.processed),
    )
    LOGGER.info(
        "Total failed files count",
        failed_count=len(state.failed),
    )
    LOGGER.info("Finished processing")


if __name__ == "__main__":
    main()
