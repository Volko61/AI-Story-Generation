"""
Simple CLI to generate a multi-chapter book using OpenAI's Chat API.

Usage (text prompt):
    python book_writer.py --prompt "write a book about sworms"

Optional speech input (requires SpeechRecognition and a working microphone):
    python book_writer.py --speech

Set your key in PowerShell before running:
    $env:OPENAI_API_KEY = "sk-..."

This module is safe to import (it won't call the API on import).
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import List

from agents import Agent, Runner
from agents.run import RunConfig
from agents.models.multi_provider import MultiProvider


def load_env_file(path: str = ".env") -> dict:
    data: dict[str, str] = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                data[k.strip()] = v
    return data


def get_run_config_from_env(env_path: str = ".env") -> RunConfig:
    env = load_env_file(env_path)
    api_key = env.get("APIKEY") or os.getenv("OPENAI_API_KEY")
    model = env.get("MODEL") or os.getenv("MODEL") or "gpt-4"
    base_url = env.get("BASE_URL") or os.getenv("BASE_URL")

    # Force the provider to use chat completions style (not Responses API) because
    # some endpoints (like OpenRouter) may not support the Responses API shape.
    provider = MultiProvider(openai_api_key=api_key, openai_base_url=base_url, openai_use_responses=False)

    # Set OPENAI_API_KEY env var for any downstream libraries that expect it.
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key
    run_config = RunConfig(model=model, model_provider=provider)

    # If we're pointing to a non-openai base URL (for example OpenRouter), disable
    # trace export so the SDK doesn't try to upload traces to platform.openai.com and
    # get 401 errors for keys that are only valid on the other endpoint.
    if base_url and "openai.com" not in base_url:
        run_config.tracing_disabled = True

    return run_config


def generate_toc(book_prompt: str, chapters: int = 5, run_config: RunConfig | None = None) -> List[str]:
    # Create a minimal agent to generate TOC using the agent SDK
    agent = Agent(
        name="TOC Agent",
        instructions=(
            "You are a helpful assistant that creates structured book outlines. "
            "Respond with a numbered table of contents with exactly the requested number of chapters."
        ),
    )

    user_text = f"Create a {chapters}-chapter table of contents for this book idea: {book_prompt}\nRespond with each chapter on its own line, numbered."

    try:
        print(f"Using model={run_config.model}, base_url={(run_config.model_provider.openai_provider._stored_base_url if hasattr(run_config.model_provider, 'openai_provider') else None)}")
        if hasattr(run_config.model_provider, 'openai_provider'):
            stored_key = getattr(run_config.model_provider.openai_provider, '_stored_api_key', None)
            if stored_key:
                print(f"API key: {stored_key[:6]}...{stored_key[-4:]}")
        result = Runner.run_sync(agent, user_text, run_config=run_config)
        text = result.final_output or ""
    except Exception as e:
        # Make the error clearer for troubleshooting
        print(f"Error running agent: {e}")
        raise
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    titles: List[str] = []
    for line in lines:
        parts = line.split(".", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            titles.append(parts[1].strip())
        else:
            titles.append(line)

    if len(titles) >= chapters:
        return titles[:chapters]
    else:
        while len(titles) < chapters:
            titles.append(f"Chapter {len(titles)+1}")
        return titles


def generate_main_idea(book_prompt: str, run_config: RunConfig | None = None) -> str:
    agent = Agent(
        name="Main Idea",
        instructions=("Summarize the main idea of the book in a clear paragraph."
                      " Keep it focused and suitable as a short blurb."),
    )
    try:
        result = Runner.run_sync(agent, f"Book idea: {book_prompt}\n\nSummarize the main idea." , run_config=run_config)
        return result.final_output or ""
    except Exception as e:
        print(f"Error generating main idea: {e}")
        raise


def generate_secondary_ideas(book_prompt: str, count: int = 5, run_config: RunConfig | None = None) -> List[str]:
    agent = Agent(
        name="Secondary Ideas",
        instructions=("List several secondary ideas, subplots, or themes that complement the main idea."
                      " Respond with each idea on its own line."),
    )
    try:
        result = Runner.run_sync(agent, f"Book idea: {book_prompt}\n\nList {count} secondary ideas, one per line.", run_config=run_config)
        text = result.final_output or ""
        lines = [l.strip('- ').strip() for l in text.splitlines() if l.strip()]
        return lines[:count]
    except Exception as e:
        print(f"Error generating secondary ideas: {e}")
        raise


def generate_characters(book_prompt: str, count: int = 6, run_config: RunConfig | None = None) -> List[str]:
    agent = Agent(
        name="Characters",
        instructions=("List important characters for the book idea. For each character, include a short description"
                      " and role in one line. Respond with one character per line."),
    )
    try:
        result = Runner.run_sync(agent, f"Book idea: {book_prompt}\n\nList {count} key characters, one per line with name and short description.", run_config=run_config)
        text = result.final_output or ""
        lines = [l.strip('- ').strip() for l in text.splitlines() if l.strip()]
        return lines[:count]
    except Exception as e:
        print(f"Error generating characters: {e}")
        raise


def generate_chapter_ideas(book_prompt: str, toc: List[str], run_config: RunConfig | None = None) -> List[str]:
    agent = Agent(
        name="Chapter Ideas",
        instructions=("For each chapter title, give a short paragraph describing the chapter's core idea and scenes."
                      " Respond with each chapter idea separated by a blank line or on separate lines starting with the chapter number."),
    )
    prompt = f"Book idea: {book_prompt}\n\nTable of contents:\n"
    for i, t in enumerate(toc, 1):
        prompt += f"{i}. {t}\n"
    prompt += "\nFor each chapter above, write a short idea/outline (1-3 sentences) prefixed by the chapter number."
    try:
        result = Runner.run_sync(agent, prompt, run_config=run_config)
        text = result.final_output or ""
        # Split into blocks per chapter by lines starting with digit.
        lines = [l for l in text.splitlines() if l.strip()]
        ideas: List[str] = []
        for line in lines:
            parts = line.split('.', 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                ideas.append(parts[1].strip())
            else:
                ideas.append(line.strip())
        return ideas[: len(toc)]
    except Exception as e:
        print(f"Error generating chapter ideas: {e}")
        raise


def generate_chapter_parts_ideas(book_prompt: str, chapter_title: str, parts: int = 3, run_config: RunConfig | None = None) -> List[str]:
    agent = Agent(
        name="Chapter Parts",
        instructions=("Break the chapter into parts/scenes. Respond with each part idea on its own line, numbered.")
    )
    prompt = f"Book idea: {book_prompt}\nChapter: {chapter_title}\n\nBreak this chapter into {parts} parts and give a short idea for each part, numbered."
    try:
        result = Runner.run_sync(agent, prompt, run_config=run_config)
        text = result.final_output or ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        parts_list: List[str] = []
        for line in lines:
            p = line.split('.', 1)
            if len(p) == 2 and p[0].strip().isdigit():
                parts_list.append(p[1].strip())
            else:
                parts_list.append(line)
        # Ensure we have exactly 'parts' entries
        while len(parts_list) < parts:
            parts_list.append(f"Part {len(parts_list)+1}")
        return parts_list[:parts]
    except Exception as e:
        print(f"Error generating chapter parts ideas: {e}")
        raise


def save_text_file(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Saved: {path}")



def generate_chapter(book_prompt: str, chapter_title: str, chapter_index: int, run_config: RunConfig | None = None) -> str:
    agent = Agent(
        name="Chapter Writer",
        instructions=(
            "You are a creative fiction writer. Write a full chapter based on the chapter title and the book idea. "
            "Aim for an engaging chapter of ~700-1200 words, consistent voice and clear scenes."
        ),
    )

    user_text = (
        f"Book idea: {book_prompt}\n\nChapter {chapter_index}: {chapter_title}\n\n"
        "Write the chapter content. Keep consistent voice across chapters. Include scene details and natural transitions."
    )

    try:
        print(f"Generating chapter {chapter_index} with model={run_config.model}")
        result = Runner.run_sync(agent, user_text, run_config=run_config)
        return result.final_output or ""
    except Exception as e:
        print(f"Error generating chapter {chapter_index}: {e}")
        raise


def save_book(title: str, chapters: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    safe_title = "_".join(title.lower().split())[:120]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    book_base = os.path.join(out_dir, f"{safe_title}_{timestamp}")
    book_file = f"{book_base}.txt"

    with open(book_file, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        for i, chap in enumerate(chapters, start=1):
            f.write(f"Chapter {i}\n")
            f.write(chap + "\n\n")

    print(f"Saved book to: {book_file}")
    return book_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a book from a short idea/prompt using the agents SDK")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", "-p", help="Text prompt describing the book idea")
    group.add_argument("--speech", "-s", action="store_true", help="Record a short speech prompt from the microphone")
    parser.add_argument("--chapters", "-n", type=int, default=5, help="Number of chapters (default 5)")
    parser.add_argument("--out", "-o", default="outputs", help="Output directory")
    parser.add_argument("--parts", "-m", type=int, default=3, help="Number of parts per chapter (default 3)")
    parser.add_argument("--env", "-e", default=".env", help="Path to .env file (default .env)")

    args = parser.parse_args(argv)

    if args.speech:
        # Lazy import speech_recognition only when needed. Use a safe import to avoid static import errors.
        try:
            import speech_recognition as sr  # type: ignore
        except Exception:
            sr = None  # type: ignore

        try:
            if sr is None:
                raise RuntimeError("speech_recognition is not installed")
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("Listening... (speak your book prompt)")
                audio = r.listen(source, phrase_time_limit=12)
            book_prompt = r.recognize_google(audio)
            print("Recognized:", book_prompt)
        except Exception as e:
            raise RuntimeError("Speech recognition failed or is not installed") from e
    else:
        book_prompt = args.prompt

    run_config = get_run_config_from_env(args.env)

    print(f"Generating TOC for: {book_prompt}")
    toc = generate_toc(book_prompt, chapters=args.chapters, run_config=run_config)
    print("Table of contents:")
    for i, t in enumerate(toc, 1):
        print(f"  {i}. {t}")

    # Create a folder for this run
    run_dir = os.path.join(args.out, f"book_{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)

    # 1) Main idea
    main_idea = generate_main_idea(book_prompt, run_config=run_config)
    save_text_file(os.path.join(run_dir, "00_main_idea.txt"), main_idea)

    # 2) Secondary ideas
    secondary = generate_secondary_ideas(book_prompt, count=8, run_config=run_config)
    sec_text = "\n".join(f"- {s}" for s in secondary)
    save_text_file(os.path.join(run_dir, "01_secondary_ideas.txt"), sec_text)

    # 3) Characters
    characters = generate_characters(book_prompt, count=8, run_config=run_config)
    char_text = "\n".join(f"- {c}" for c in characters)
    save_text_file(os.path.join(run_dir, "02_characters.txt"), char_text)

    # 4) Chapter ideas
    chapter_ideas = generate_chapter_ideas(book_prompt, toc, run_config=run_config)
    chap_ideas_text = "\n\n".join(f"Chapter {i+1}: {idea}" for i, idea in enumerate(chapter_ideas))
    save_text_file(os.path.join(run_dir, "03_chapter_ideas.txt"), chap_ideas_text)

    # 5) Parts ideas per chapter
    parts_per_chapter: list[list[str]] = []
    for i, title in enumerate(toc, 1):
        parts_ideas = generate_chapter_parts_ideas(book_prompt, title, parts=args.parts, run_config=run_config)
        parts_per_chapter.append(parts_ideas)
        parts_text = "\n".join(f"{idx+1}. {p}" for idx, p in enumerate(parts_ideas))
        save_text_file(os.path.join(run_dir, f"04_chapter_{i:02d}_parts_ideas.txt"), parts_text)

    # 6) Generate each part content and save separately
    chapter_parts_paths: list[list[str]] = []
    for i, title in enumerate(toc, 1):
        part_paths: list[str] = []
        for j, part_idea in enumerate(parts_per_chapter[i-1], 1):
            print(f"Generating chapter {i} part {j}: {title} - {part_idea[:40]}")
            # Use the chapter writer agent but provide the part-specific prompt
            agent = Agent(
                name="Chapter Part Writer",
                instructions=(
                    "You are a fiction writer. Write a scene/part of a chapter based on the chapter title, the part idea, and the book idea."
                ),
            )
            user_text = (
                f"Book idea: {book_prompt}\nChapter {i}: {title}\nPart {j} idea: {part_idea}\n\nWrite a coherent scene of ~400-900 words for this part."
            )
            try:
                result = Runner.run_sync(agent, user_text, run_config=run_config)
                part_text = result.final_output or ""
            except Exception as e:
                print(f"Error generating chapter {i} part {j}: {e}")
                part_text = f"[Error generating part {j} of chapter {i}]"

            part_file = os.path.join(run_dir, f"chapter_{i:02d}_part_{j:02d}.txt")
            save_text_file(part_file, part_text)
            part_paths.append(part_file)
        chapter_parts_paths.append(part_paths)

    # 7) Assemble each chapter from its parts
    chapter_files: list[str] = []
    for i, part_paths in enumerate(chapter_parts_paths, 1):
        chapter_text = f"{toc[i-1]}\n\n"
        for j, pth in enumerate(part_paths, 1):
            with open(pth, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            chapter_text += f"Part {j}\n\n{content}\n\n"
        chapter_file = os.path.join(run_dir, f"chapter_{i:02d}.txt")
        save_text_file(chapter_file, chapter_text)
        chapter_files.append(chapter_file)

    # 8) Glue all chapters into final book file with formatting
    all_chapters = []
    book_title = book_prompt
    book_header = f"{book_title}\nGenerated: {datetime.now().isoformat()}\n\n"
    for i, ch_file in enumerate(chapter_files, 1):
        with open(ch_file, 'r', encoding='utf-8') as f:
            ch_text = f.read().strip()
        all_chapters.append(f"Chapter {i}: {toc[i-1]}\n\n{ch_text}")

    final_book_text = book_header + "\n\n".join(all_chapters)
    final_book_path = os.path.join(run_dir, f"book_complete.txt")
    save_text_file(final_book_path, final_book_text)

    print(f"All done. Files saved under: {run_dir}")



if __name__ == "__main__":
    main()
