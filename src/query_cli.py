from rag import build_rag_chain, load_existing_vector_store


def main():
    print("\nRAG Interactive Query CLI")
    print("=" * 50)
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    vector_store = load_existing_vector_store()
    chain        = build_rag_chain(vector_store)

    while True:
        question = input("Your question: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not question:
            continue

        result = chain.invoke({"input": question})
        print(f"\nAnswer:\n{result['answer']}")
        sources = list({d.metadata.get("source", "N/A") for d in result["context"]})
        print(f"\nSources: {sources}\n")
        print("-" * 50)


if __name__ == "__main__":
    main()