import argparse
from rag import Ingestor, RAGSystem


def main():
    parser = argparse.ArgumentParser(description="Minimal RAG CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Ingest documents from a directory")
    p_ingest.add_argument("--source_dir", required=True)
    p_ingest.add_argument("--index_path", default="./index.faiss")
    p_ingest.add_argument("--chunk_size", type=int, default=150)
    p_ingest.add_argument("--overlap", type=int, default=25)

    p_query = sub.add_parser("query", help="Query the RAG system")
    p_query.add_argument("--k", type=int, default=3)
    p_query.add_argument("--index_path", default="./index.faiss")
    p_query.add_argument("query", nargs="+", help="Question to ask")

    p_debug = sub.add_parser("debug", help="Debug the index")
    p_debug.add_argument("--index_path", default="./index.faiss")
    p_debug.add_argument("search_term", help="Search term to find in chunks")

    args = parser.parse_args()

    if args.cmd == "ingest":
        ing = Ingestor(
            args.source_dir,
            index_path=args.index_path,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        ing.ingest()

    elif args.cmd == "query":
        q = " ".join(args.query)
        rag = RAGSystem(index_path=args.index_path)
        print(rag.answer(q, k=args.k))

    elif args.cmd == "debug":
        rag = RAGSystem(index_path=args.index_path)

        print(f"Searching for chunks containing: '{args.search_term}'")
        print("=" * 50)

        matches = []
        for i, text in enumerate(rag.texts):
            if args.search_term.lower() in text.lower():
                matches.append((i, text))

        if matches:
            print(f"Found {len(matches)} matching chunks:")
            for idx, text in matches:
                metadata = rag.metadatas[idx]
                print(f"\n--- Chunk {idx} ---")
                print(f"Source: {metadata['source']} (chunk {metadata['chunk']})")
                print(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}")
        else:
            print("No chunks found containing the search term!")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()