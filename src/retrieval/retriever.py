import numpy as np

class Retriever:
    def __init__(self, embeddings, documents):
        self.embeddings = embeddings
        self.documents = documents

    def search(self, query_embedding, top_k: int = 3):
        
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for indices in top_indices:
            results.append(
                {
                    "document": self.documents[indices],
                    "score": float(scores[indices])
                }
            )
        return results

