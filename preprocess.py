import faiss
import pandas as pd
import numpy as np
from helper import clean
from sentence_transformers import SentenceTransformer


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
    print("Created {} query-response pairs.".format(len(pairs)))

    # Split pairs into train/dev/test
    n = len(pairs)
    training_size = int(n * 0.8)
    dev_size = int(n * 0.9)

    training_pairs = pairs[:training_size]
    dev_pairs = pairs[training_size:dev_size]
    test_pairs = pairs[dev_size:]

    training_df = pd.DataFrame(training_pairs)
    dev_df = pd.DataFrame(dev_pairs)
    test_df = pd.DataFrame(test_pairs)
    print("Training shape:", training_df.shape)
    print("Dev shape:", dev_df.shape)
    print("Test shape:", test_df.shape)

    print("Writing train/dev/test as CSV files...")
    training_df.to_csv(
        f"artifacts/datasets/training.csv",
        index=False,
    )
    dev_df.to_csv(
        f"artifacts/datasets/dev.csv",
        index=False,
    )
    test_df.to_csv(
        f"artifacts/datasets/test.csv",
        index=False,
    )
