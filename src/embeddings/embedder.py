import os

# Limit numerical libraries to a small number of threads.
# This helps prevent OpenBLAS/OMP memory allocation problems.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            model_name
        )

        print("Embedding model loaded.")

    def embed_documents(
        self,
        documents: list[str]
    ):

        return self.model.encode(
            documents,
            normalize_embeddings=True,
            batch_size=8,
            show_progress_bar=True
        )

    def embed_query(
        self,
        query: str
    ):

        return self.model.encode(
            query,
            normalize_embeddings=True
        )