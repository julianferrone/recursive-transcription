# recursive-transcription

Recursively transcribe audio files into text.

## Setup

```python
pip install structlog
pip install -U openai-whisper
```

## Usage

This script transcribes audio files into text using the Whisper model. The command-line interface (CLI) allows you to configure various transcription options, specify file paths, and control the level of verbosity. Below are the available arguments and their descriptions:

`python transcribe_for_llm.py \[OPTIONS\] paths`

### Options

| Name | Type | Description | Default |
| --- | --- | ---| --- |
| `-m`, `--model` | string | The Whisper model name. | `base` |
| `-t`, `--transcript-dir` | string | Specifies the directory to save the transcription files. If not provided, transcript files will be saved next to the audio file. If the flag is provided without an argument, transcripts will be saved in a folder named transcripts under the current directory. | `./transcripts` |
| `--include-timestamps` | boolean | Whether to include timestamps in the transcribed text files. | `False` |
| `-v`, `--verbose` | int | Controls the level of verbosity for the transcription process. No flag: No progress bar. One -v: Small progress display. Two -vs (e.g., -vv): Logs the audio transcripts to the console. | `0` |
| `--ignore-existing` | bool | If set to True, transcription will skip files that already have an existing transcript. | `True` |
| `--log-level` | Choice of `DEBUG`, `INFO`, `WARNING`, `ERROR` | Changes the logging level of the script. | `INFO` | 

### Required

`paths` is a list of one or more paths to directories or individual files to be transcribed. Directories will have all descendant audio files transcribed.

### Example Usage

| Usage | Command |
| --- | --- |
| Transcribe audio files in a directory with timestamps | `python transcribe.py --include-timestamps /path/to/audio/files` |
| Transcribe audio files with the large Whisper model and log progress to the console | `python transcribe.py -m large -vv /path/to/audio/files` | 
| Transcribe files, without skipping already existing transcriptions | `python transcribe.py --no-ignore-existing /path/to/audio/files` |
| Save transcriptions to a specific directory | `python transcribe.py -t /path/to/save/transcripts /path/to/audio/files` |
