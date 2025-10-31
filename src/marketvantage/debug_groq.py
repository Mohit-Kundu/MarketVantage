import sys
from marketvantage.llm_groq import generate_answer


def main() -> None:
    question = "Say 'hello' and count to three."
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    context = [
        "This is a test context. You can greet and count to three: one, two, three.",
        "Feel free to answer concisely.",
    ]
    ans = generate_answer(question, context, max_output_tokens=128)
    print(f"OK Groq. Response length: {len(ans)}")
    print("Snippet:\n" + (ans[:500] if ans else "<empty>"))


if __name__ == "__main__":
    main()
