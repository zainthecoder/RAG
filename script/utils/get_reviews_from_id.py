import json
import pickle
import gzip
from transformers import pipeline
import random
from sentiment_analysis import predict_sentiment
import os

# pipeline for sentiment analysis
sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
MAX_TEXT_SIZE_THRESHOLD = 128  


### load the data
def parse(path):
    data = []
    with gzip.open(path) as f:
        for l in f:
            data.append(json.loads(l.strip()))
        return(data)

def load_json(file_path):
    """Load data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.")
        return []
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# Load unique IDs from pickle file
with open('/home/stud/abedinz1/localDisk/RAG/RAG/data/unique_ids.pickle', 'rb') as f:
    unique_ids = pickle.load(f)
    #print(unique_ids)

# Function to generate asin_reviewerID key
def generate_key(asin, reviewerID):
    return f"{asin}_{reviewerID}"

# Decompress reviews JSON file
#reviews =  parse('/home/stud/abedinz1/localDisk/RAG/RAG/data/Cell_Phones_and_Accessories_5.json.gz')
reviews = load_json('/home/stud/abedinz1/localDisk/RAG/RAG/data/filtered_reviews.json')

# Filter reviews based on unique IDs
filtered_reviews = []
unique_product_ids = set()
for review in reviews:
    key = generate_key(review['asin'], review['reviewerID'])
    if key in unique_ids:
        review_text = review.get('reviewText', " ")
        print(predict_sentiment(review_text))
        unique_product_ids.append(review["asin"])
        
        # # Check if the text size exceeds the threshold
        # if len(review_text) > MAX_TEXT_SIZE_THRESHOLD:
        #     # Assign random sentiment if text size is too large
        #     review["sentiment"] = random.choice(["POSITIVE_random", "NEGATIVE_random"])
        # else:
        #     # Run sentiment analysis model
        #     sentiment_result = sentiment_pipeline(review_text)
        #     review["sentiment"] = sentiment_result[0]["label"]
        # print(review)
        # filtered_reviews.append(review)

# Save filtered reviews to a new JSON file
with open('/home/stud/abedinz1/localDisk/RAG/RAG/data/filtered_reviews.json', 'w') as f:
    json.dump(filtered_reviews, f, indent=4)


# File path to save the Python list
file_path = "../../data/unique_product_ids.pickle"


# Open the file in binary mode
with open(file_path, 'wb') as file:
    # Serialize and write the variable to the file
    pickle.dump(unique_product_ids, file)