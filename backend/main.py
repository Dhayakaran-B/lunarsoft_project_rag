from retrieve import retrieve
from generate import generate_answer


def ask_question(question):

    print("\nSearching the document...")

    context = retrieve(question)

    print(f"Retrieved chunks: {len(context)}")

    if not context:
        return {
            "answer": "I don't know based on the provided document.",
            "sources": []
        }

    print("\nRetrieved evidence:")

    for i, item in enumerate(context, start=1):

        print(
            f"\n[{i}] Page {item['page']} "
            f"| Retrieval score: {item['score']:.4f}"
        )

        print(
            item["text"][:300]
            .replace("\n", " ")
        )

    answer = generate_answer(
        question,
        context
    )

    sources = sorted(
        set(
            item["page"]
            for item in context
        )
    )

    return {
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":

    print("=" * 60)
    print("PDF RAG TEST SYSTEM")
    print("=" * 60)

    while True:

        question = input(
            "\nAsk a question (or type 'exit'): "
        )

        if question.lower() == "exit":
            break

        result = ask_question(question)

        print("\n")
        print("=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(result["answer"])

        print("\nRetrieved pages:")
        print(result["sources"])