# AI-Story-Generation
# Book Writer

Simple CLI to generate a multi-chapter book using OpenAI's Chat API.

Prerequisites
- Python 3.8+ and a working internet connection.
- Set your OpenAI API key in PowerShell before running:

```
$env:OPENAI_API_KEY = "sk-..."
```

Install dependencies (recommended inside a virtual environment):

```
pip install -r requirements.txt
```

Usage examples

- Text prompt:

```
python book_writer.py --prompt "write a book about sworms" --chapters 6
```

- Speech prompt (requires microphone and SpeechRecognition installed):

```
python book_writer.py --speech --chapters 4
```

Output
- The script saves a text file under `outputs/` containing the book.

Notes
- The script imports optional packages (openai, speech_recognition) at runtime so the module is safe to import without installing all deps.
