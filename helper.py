import faiss
import re
from sentence_transformers import SentenceTransformer


def clean(text: str) -> str:
    """
    Clean the text by:
    - Removing URLs
    - Removing leading mentions
    - Normalizing placeholders
    - Stripping whitespace
    """
    text = re.sub(r"http\S+", "", text) # or re.sub(r"http\S+", "[URL]", text)
    text = re.sub(r"^@\w+\s*", "", text)
    text = text.replace("__email__", "[email]")
    text = text.replace("__phone__", "[phone]")

    return text.strip()


def batch_iter(data: list, batch_size: int) -> list:
    """
    Batch the data into smaller lists of the given size.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def build_index(embed_dim: int) -> faiss.Index:
    """
    Build a FAISS index for cosine similarity via inner product.
    """
    index = faiss.IndexFlatIP(embed_dim)

    return index


def build_faiss_index(docs: list, model: SentenceTransformer, index: faiss.Index, batch_size: int):
    """
    Build a FAISS index for cosine similarity via inner product from a list of documents.
    """
    for batch in batch_iter(docs, batch_size):
        
        texts = [d["embedding_text"] for d in batch]

        # embedding
        emb = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype("float32")

        # normalize for cosine similarity
        faiss.normalize_L2(emb)

        # add to index (NO storing embeddings in RAM)
        index.add(emb)
    
    print("Embedded shape:", emb.shape)
    print("Indexed {} documents.".format(index.ntotal))

    return index


def search(query: str, index: faiss.Index, model: SentenceTransformer, docs: list, k: int = 5) -> list:
    """ 
    Return the top k most similar documents to the query.
    """
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    distances, indices = index.search(q_emb, k)

    results = []
    for idx in indices[0]:
        results.append(docs[idx])

    return results



