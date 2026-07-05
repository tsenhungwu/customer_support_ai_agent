import faiss
import pandas as pd
import numpy as np
from helper import clean, build_index, build_faiss_index, search
from sentence_transformers import SentenceTransformer

TRAINING_SIZE = 600000 # number of training samples
BATCH_SIZE = 512 # batch size for embedding
EMBED_DIM = 384 # embedding dimension

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def build_pipeline(docs):
    """
    Build the pipeline for the customer support AI agent.
    """
    index = build_index(EMBED_DIM)
    index = build_faiss_index(docs, MODEL, index, BATCH_SIZE)
    return index


if __name__ == "__main__":

    # Read in data
    print("Loading data...")
    data = pd.read_csv("../data/customer-support-on-twitter/twcs/twcs.csv")
    print("Loaded data with shape:", data.shape)

    # Preprocess raw data
    print("Cleaning data...")
    data["text"] = data["text"].apply(clean)
    data = data[data["text"] != ""]
    print("Cleaned data with shape:", data.shape)

    # Join question with response
    first_inbound = data[pd.isnull(data.in_response_to_tweet_id) & data.inbound]
    print("Found {} first inbound messages.".format(len(first_inbound)))

    # Merge in all tweets in response
    inbounds_and_outbounds = pd.merge(first_inbound, data, left_on="tweet_id", right_on="in_response_to_tweet_id")
    print("Found {} responses.".format(len(inbounds_and_outbounds)))

    # Filter out cases where reply tweet isn't from company
    inbounds_and_outbounds = inbounds_and_outbounds[inbounds_and_outbounds.inbound_y ^ True]
    print("Found {} responses from companies.".format(len(inbounds_and_outbounds)))

    # customer tweet → reply tweet
    print("Creating query-response pairs...")
    pairs = []
    for _, row in inbounds_and_outbounds.iterrows():
        author_id = row["author_id_x"]
        query = row["text_x"]
        response = row["text_y"]
    
        pairs.append({
            "author_id": author_id,
            "query_text": query,
            "response_text": response,
            "embedding_text": f"customer: {query} \nsupport: {response}"
        })

    training_pairs = pairs[:TRAINING_SIZE]
    print("Created {} query-response pairs.".format(len(pairs)))
    print("Selected {} training pairs.".format(len(training_pairs)))

    print("Building pipeline...")
    index = build_pipeline(training_pairs)

    testing_query = "I can't login to my account"
    print("Query:", testing_query)
    
    print("Searching for answers...")
    results = search(testing_query, index, MODEL, training_pairs, k=5)
    for r in results:
        print("Q:", r["query_text"])
        print("A:", r["response_text"])
        print("-" * 40)