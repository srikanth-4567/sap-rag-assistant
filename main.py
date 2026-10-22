from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_text
from src.embeddings.embedder import Embedder
from src.retrieval.retriever import Retriever

PDF_PATH = "data/raw/sap/S4D437_EN_Col24_CO_A4.pdf"


def main():

    pages = load_pdf(PDF_PATH)

    print(f"Pages extracted: {len(pages)}")

    # -----------------------------
    # 2. Chunk each page
    # -----------------------------

    all_chunks = []

    for page in pages:

        text = page["text"]

        if not text:
            continue

        chunks = chunk_text(
            text,
            chunk_size=500,
            overlap=100,
        )

        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            all_chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": page["metadata"]["source"],
                        "page": page["metadata"]["page"],
                        "chunk_id": (
                            f"p{page['metadata']['page']}"
                            f"_c{chunk_number}"
                        ),
                    },
                }
            )

 # -----------------------------
    # 3. Create embeddings
    # -----------------------------

    embedder = Embedder()

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = embedder.embed_documents(
        texts
    )

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    # -----------------------------
    # 3. Statistics
    # -----------------------------

    print(
        f"Total chunks created: "
        f"{len(all_chunks)}"
    )

    # -----------------------------
    # 4. Create retriever
    # -----------------------------
    retriever = Retriever(
        embeddings,
        all_chunks
    )

    # -----------------------------
    # 5. User query
    # -----------------------------
    queries = [
    "How is EML used in RAP?",
    "What are the objectives of the EML lesson?",
    "What is RAP architecture?",
    "What topics are covered in draft-enabled services?",
    "What are the extensibility options in RAP?",
    ]

    for query in queries:
        print("\n")
        print("#" * 100)
        print(f"QUERY: {query}")
        print("#" * 100)
        query_embedding = embedder.embed_query(
            query
        )

        results = retriever.search(
            query_embedding,
            top_k=3
            )

        for rank, result in enumerate(
            results,
            start = 1
        ):
            document = result["document"]
            print("\n" + "=" * 80)
            print(f"Rank: {rank}")
            print(f"Score: {result['score']:.4f}")
            print(
                f"Page: "
                f"{document['metadata']['page']}"
            )
            print(
                f"Chunk: "
                f"{document['metadata']['chunk_id']}"
            )
            print("\nText:")
            print(document["text"][:500])

    print(
        f"\nUser query: {query}"
    )
'''
    # -----------------------------
    # 6. Embed query
    # -----------------------------

    query_embedding = embedder.embed_query(
        query
    )
    # -----------------------------
    # 7. Retrieve top 3
    # -----------------------------

    results = retriever.search(
        query_embedding,
        top_k=3
    )
'''
    # -----------------------------
    # 8. Display results
    # -----------------------------
'''    print("\nTop retrieved chunks:")

    for rank, result in enumerate(results, start=1):
        document = result["document"]
        print("\n" + "=" * 80)
        print(f"Rank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(
            f"Page: {document['metadata']['page']}"
        )
        print(
            f"Chunk ID: "
            f"{document['metadata']['chunk_id']}"
        )
        print("\nText:")
        print(document["text"][:1500])
'''

    # -----------------------------
    # 4. Display first 5 chunks
    # -----------------------------
'''
    for chunk in all_chunks[:5]:

        print("\n" + "=" * 80)

        print(
            f"Source: "
            f"{chunk['metadata']['source']}"
        )

        print(
            f"Page: "
            f"{chunk['metadata']['page']}"
        )

        print(
            f"Chunk ID: "
            f"{chunk['metadata']['chunk_id']}"
        )

        print("\nText:")

        print(chunk["text"][:1000])

        " to fetch the content of 3 pages"


    print("\n" + "=" * 80)

    for document in pages[2:5]:

        print(
            f"Source: {document['metadata']['source']}"
        )

        print(
            f"Page: {document['metadata']['page']}"
        )

        print("\nText:")

        print(document["text"][:1000])

        print("\n" + "=" * 80)
'''

if __name__ == "__main__":
    main()