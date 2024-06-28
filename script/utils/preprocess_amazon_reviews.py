import json
import pickle
import gzip
import random
import os
from transformers import pipeline
from pyabsa import AspectTermExtraction as ATEPC, DeviceTypeOption
import pprint

# Constants
UNIQUE_IDS_PATH = '/home/stud/abedinz1/localDisk/RAG/RAG/data/unique_ids.pickle'

#REVIEWS_PATH = '/home/stud/abedinz1/localDisk/RAG/RAG/data/filtered_reviews.json'
REVIEWS_PATH =  '/home/stud/abedinz1/localDisk/RAG/RAG/data/Cell_Phones_and_Accessories_5.json.gz'

OUTPUT_REVIEWS_PATH = '/home/stud/abedinz1/localDisk/RAG/RAG/data/filtered_reviews.json'
UNIQUE_PRODUCT_IDS_PATH = "../../data/unique_product_ids.pickle"

def setup_aspect_extractor(language='english'):
    """Initialize the Aspect Term Extraction model with the specified language."""
    return ATEPC.AspectExtractor(language, auto_device=DeviceTypeOption.AUTO)

# Initialize the aspect extractor
aspect_extractor = setup_aspect_extractor('english')

def parse_gzip_json(path):
    """Parse a gzip compressed JSON file."""
    data = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def load_json(file_path):
    """Load data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.")
        return []
    with open(file_path, "r") as f:
        return json.load(f)

def generate_key(asin, reviewerID):
    """Generate a unique key from asin and reviewerID."""
    return f"{asin}_{reviewerID}"

def load_unique_ids(path):
    """Load unique IDs from a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def save_pickle(data, file_path):
    """Save data to a pickle file."""
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)

def main():
    unique_ids = load_unique_ids(UNIQUE_IDS_PATH)
    
    #reviews = load_json(REVIEWS_PATH)
    reviews = parse_gzip_json(REVIEWS_PATH)

    filtered_reviews = []
    unique_product_ids = set()
    print(filtered_reviews)

    for review in reviews:
        key = generate_key(review['asin'], review['reviewerID'])
        if key in unique_ids:
            review_text = review.get('reviewText', " ")

            result = aspect_extractor.predict(
                [review_text],
                save_result=False,
                print_result=True,
                ignore_error=True
            )

            review['aspect'] = result[0]['aspect']
            review['sentiment'] = result[0]['sentiment']
            filtered_reviews.append(review)
            unique_product_ids.add(review["asin"])

    save_json(filtered_reviews, OUTPUT_REVIEWS_PATH)
    save_pickle(unique_product_ids, UNIQUE_PRODUCT_IDS_PATH)
    pprint.pprint(filtered_reviews)

if __name__ == "__main__":
    main()
