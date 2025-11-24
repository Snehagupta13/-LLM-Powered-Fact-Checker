import argparse
import json
from pathlib import Path
from typing import List

from src.pipeline import run_fact_check


def read_text_from_file(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="LLM-powered fact checker CLI.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Input text to fact-check.")
    group.add_argument("--file", type=str, help="Path to a text file to fact-check.")

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of evidence facts to retrieve per claim.",
    )

    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        path = Path(args.file)
        if not path.exists():
            raise FileNotFoundError(path)
        text = read_text_from_file(path)

    result = run_fact_check(text=text, top_k=args.top_k)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
