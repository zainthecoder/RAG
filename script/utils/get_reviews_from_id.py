import json
import pickle
import gzip

### load the data
def parse(path):
    data = []
    with gzip.open(path) as f:
        for l in f:
            data.append(json.loads(l.strip()))
        return(data)


# Load unique IDs from pickle file
with open('/home/stud/abedinz1/localDisk/RAG/RAG/data/unique_ids.pickle', 'rb') as f:
    unique_ids = pickle.load(f)
    print(unique_ids)

# Function to generate asin_reviewerID key
def generate_key(asin, reviewerID):
    return f"{asin}_{reviewerID}"

# Decompress reviews JSON file
reviews =  parse('/home/stud/abedinz1/localDisk/RAG/RAG/data/Cell_Phones_and_Accessories_5.json.gz')


# Filter reviews based on unique IDs
filtered_reviews = []
for review in reviews:
    key = generate_key(review['asin'], review['reviewerID'])
    if key in unique_ids:
        filtered_reviews.append(review)

# Save filtered reviews to a new JSON file
with open('/home/stud/abedinz1/localDisk/RAG/RAG/data/filtered_reviews.json', 'w') as f:
    json.dump(filtered_reviews, f, indent=4)
