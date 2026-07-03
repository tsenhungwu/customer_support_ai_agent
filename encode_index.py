import pandas as pd
from helper import build_index, build_faiss_index
from sentence_transformers import SentenceTransformer


BATCH_SIZE = 100000 # batch size for embedding in each iteration
EMBED_DIM = 384 # embedding dimension

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def build_pipeline(docs: list[dict]):
    """
    Build the pipeline for the customer support AI agent.
    """
    index = build_index(EMBED_DIM)
    index = build_faiss_index(docs, MODEL, index, BATCH_SIZE)
    return index


if __name__ == "__main__":
    
    # Read in training set
    training_pairs = pd.read_csv("artifacts/datasets/training.csv").to_dict("records")

    # Encode training_pairs in batches, index at the end, and write them out
    build_pipeline(training_pairs)



