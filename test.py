import pandas as pd
from helper import search
from sentence_transformers import SentenceTransformer
import faiss


MODEL = SentenceTransformer("all-MiniLM-L6-v2")

TESTING_QUERY = "I can't login to my account"
K = 5


if __name__ == "__main__":
    
    # Read in training set
    training_pairs = pd.read_csv("artifacts/datasets/training.csv").to_dict("records") 

    print("Query:", TESTING_QUERY)
    index = faiss.read_index("artifacts/indexes/training.index")
    
    print("Recommending top {} QA pairs...".format(K))
    results = search(TESTING_QUERY, index, MODEL, training_pairs, k=K)
    for r in results:
        print("Q:", r["query_text"])
        print("A:", r["response_text"])
        print("-" * 168)
