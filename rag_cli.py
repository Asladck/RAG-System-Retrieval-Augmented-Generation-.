import argparse
from rag import Ingestor, RAGSystem


def main():
    parser = argparse.ArgumentParser(description="Minimal RAG CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Ingest documents from a directory")
    p_ingest.add_argument("--source_dir", required=True)
    p_ingest.add_argument("--index_path", default="./index.faiss")

    p_query = sub.add_parser("query", help="Query the RAG system")
    p_query.add_argument("--k", type=int, default=3)
    p_query.add_argument("--index_path", default="./index.faiss")
    p_query.add_argument("query", nargs="+", help="Question to ask")

    args = parser.parse_args()

    if args.cmd == "ingest":
        ing = Ingestor(args.source_dir, index_path=args.index_path)
        ing.ingest()
    elif args.cmd == "query":
        q = " ".join(args.query)
        rag = RAGSystem(index_path=args.index_path)
        print(rag.answer(q, k=args.k))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

