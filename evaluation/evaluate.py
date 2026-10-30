import json
import sys
from pathlib import Path


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.rag import ask_question


QUESTIONS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)


def load_questions():

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:

        return json.load(file)


def evaluate_retrieval(retrieved_documents,expected_sources):

    if not expected_sources:

        return len(retrieved_documents) == 0

    retrieved_sources = set()

    for document in retrieved_documents:

        source = document.metadata.get(
            "source",
            ""
        )

        retrieved_sources.add(source)

    return any(
        source in retrieved_sources
        for source in expected_sources
    )


def evaluate():

    questions = load_questions()

    total = len(questions)

    retrieval_correct = 0

    print("=" * 70)
    print("RAG EVALUATION")
    print("=" * 70)

    for index, item in enumerate(questions, start=1):

        question = item["question"]

        expected_sources = item["expected_sources"]

        answerable = item["answerable"]

        print()
        print(f"[{index}/{total}] {question}")

        answer, documents = ask_question(question)

        retrieval_ok = evaluate_retrieval(documents,expected_sources)

        if retrieval_ok:

            retrieval_correct += 1

            status = "PASS"

        else:

            status = "FAIL"

        print(f"Retrieval: {status}")

        print("Expected:", expected_sources)
        print("Answerable:", answerable)

        retrieved_sources = sorted(
            set(
                document.metadata.get("source", "Unknown")
                for document in documents
            )
        )

        print("Retrieved:", retrieved_sources)

        print("Answer:", answer[:300].replace("\n"," "))

    retrieval_accuracy = (retrieval_correct / total) * 100

    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(f"Total questions: {total}")

    print(f"Correct retrievals: "
          f"{retrieval_correct}"
    )

    print(f"Retrieval accuracy: "
          f"{retrieval_accuracy:.2f}%"
    )

    print("=" * 70)


if __name__ == "__main__":

    evaluate()