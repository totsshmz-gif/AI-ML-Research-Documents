import json
import sys
from pathlib import Path


#*****************
# Project root   *
#*****************

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.rag import ask_question


#********************
# Evaluation file   *
#********************

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "questions.json"
)


#****************************
# Load evaluation questions *
#****************************

def load_questions():

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


#****************************
# Check expected keywords   *
#****************************

def keyword_score(answer, expected_keywords):

    if not expected_keywords:
        return 1.0

    answer_lower = answer.lower()

    matched = 0

    for keyword in expected_keywords:

        if keyword.lower() in answer_lower:
            matched += 1

    return matched / len(expected_keywords)


#******************************************
# Check whether answer refused correctly  *
#******************************************

def refusal_check(answer):

    refusal_phrase = (
        "do not contain enough information"
    )

    answer_lower = answer.lower()

    return refusal_phrase in answer_lower


#********************
# Main evaluation   *
#********************

def evaluate():

    questions = load_questions()

    total = len(questions)

    answerable_total = 0
    answerable_pass = 0

    unanswerable_total = 0
    unanswerable_pass = 0

    keyword_scores = []
    failed_questions = []

    print("=" * 70)
    print("RAG ANSWER-QUALITY EVALUATION")
    print("=" * 70)

    for index, item in enumerate(questions, start=1):

        question = item["question"]

        expected_keywords = item["expected_keywords"]

        answerable = item["answerable"]

        print()
        print(f"[{index}/{total}] {question}")

        #************
        # Run RAG   *
        #************

        answer, documents = ask_question(question)

        #************************
        # Answerable question   *
        #************************

        if answerable:

            answerable_total += 1

            score = keyword_score(answer, expected_keywords)

            keyword_scores.append(score)

            print(f"Keyword coverage: "f"{score * 100:.1f}%")

            if score >= 0.50:

                answerable_pass += 1

                print( "Answer quality: PASS")

            else:

                print("Answer quality: FAIL")
                print("Expected keywords:")
                print(expected_keywords)
                print("Full answer:")
                print(answer)
                print("Retrieved sources:")
                for document in documents:

                    print(
                        f"  - "
                        f"{document.metadata.get('source', 'Unknown')} "
                        f"Page "
                        f"{document.metadata.get('page', 'Unknown')}"
                    )
                failed_questions.append(
                    {"question": question,
                     "score": score,
                     "expected_keywords": expected_keywords,
                     "answer": answer,
                    }
                )

        #**************************
        # Unanswerable question   *
        #**************************

        else:

            unanswerable_total += 1

            if refusal_check(answer):

                unanswerable_pass += 1

                print("Out-of-knowledge handling: PASS")

            else:

                print("Out-of-knowledge handling: FAIL")

        #****************
        # Print answer  *
        #****************

        print("Answer:", answer[:500].replace("\n"," "))

    #*********************
    # Calculate results  *
    #*********************

    if answerable_total > 0:

        answerable_accuracy = (answerable_pass/ answerable_total) * 100

    else:

        answerable_accuracy = 0

    if unanswerable_total > 0:

        refusal_accuracy = (unanswerable_pass/ unanswerable_total) * 100

    else:

        refusal_accuracy = 0

    if keyword_scores:

        average_keyword_coverage = (sum(keyword_scores)/ len(keyword_scores)) * 100

    else:

        average_keyword_coverage = 0

    #***************** 
    # Final results  *
    #*****************

    print()
    print("=" * 70)
    print("ANSWER-QUALITY RESULTS")
    print("=" * 70)

    print(
        f"Total questions: "
        f"{total}"
    )

    print(
        f"Answerable questions: "
        f"{answerable_total}"
    )

    print(
        f"Answerable questions passed: "
        f"{answerable_pass}"
    )

    print(
        f"Answerable accuracy: "
        f"{answerable_accuracy:.2f}%"
    )

    print(
        f"Average keyword coverage: "
        f"{average_keyword_coverage:.2f}%"
    )

    print(
        f"Unanswerable questions: "
        f"{unanswerable_total}"
    )

    print(
        f"Correct refusals: "
        f"{unanswerable_pass}"
    )

    print(
        f"Refusal accuracy: "
        f"{refusal_accuracy:.2f}%"
    )

    print("=" * 70)
    print()

    print("=" * 70)
    print("FAILED ANSWERABLE QUESTIONS")
    print("=" * 70)

    if failed_questions:

        for item in failed_questions:

            print()
            print(
                 f"Question: {item['question']}"
             )

            print(
                f"Keyword coverage: "
                f"{item['score'] * 100:.2f}%"
            )

            print(
                f"Expected keywords: "
                f"{item['expected_keywords']}"
            )

            print(
                f"Answer: "
                f"{item['answer']}"
            )

    else:

        print("No failed answerable questions.")

    print("=" * 70)


#***************
# Entry point  *
#***************

if __name__ == "__main__":

    evaluate()