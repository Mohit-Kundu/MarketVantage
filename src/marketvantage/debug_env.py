import os
from pathlib import Path

try:
    from dotenv import load_dotenv, find_dotenv
except Exception:
    load_dotenv = None  # type: ignore
    find_dotenv = None  # type: ignore


def mask(key: str | None) -> str:
    if not key:
        return "<missing>"
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def main() -> None:
    # Attempt to load .env from cwd or repo root
    if load_dotenv is not None and find_dotenv is not None:
        p = find_dotenv(usecwd=True)
        if p:
            load_dotenv(dotenv_path=p, override=False)
            print(f"Loaded .env via find_dotenv: {p}")
    repo_env = Path(__file__).resolve().parents[2] / ".env"
    if load_dotenv is not None and repo_env.exists():
        load_dotenv(dotenv_path=str(repo_env), override=False)
        print(f"Loaded .env via repo path: {repo_env}")

    groq = os.getenv("GROQ_API_KEY")
    you = os.getenv("YOU_API_KEY") or os.getenv("YOUCOM_API_KEY")

    print(f"GROQ_API_KEY present: {bool(groq)} length: {len(groq) if groq else 0} value: {mask(groq)}")
    print(f"YOU_API_KEY/YOUCOM_API_KEY present: {bool(you)} length: {len(you) if you else 0} value: {mask(you)}")


if __name__ == "__main__":
    main()

