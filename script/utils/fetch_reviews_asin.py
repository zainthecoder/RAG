import gzip
import json

# Function to load JSON data from a gzipped file
def parse(path):
    data = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Function to filter documents based on the "asin" field
def filter_documents_by_asin(documents, target_asin):
    return [doc for doc in documents if doc.get('asin') == target_asin]

# Path to the JSON gzipped file
file_path = '/home/stud/abedinz1/localDisk/RAG/RAG/Cell_Phones_and_Accessories_5.json.gz'

# Target ASIN value to filter documents
target_asin = "B00836Y6B2"

# Load JSON data from the gzipped file
data = parse(file_path)

# Filter documents based on the target ASIN value
filtered_documents = filter_documents_by_asin(data, target_asin)

# File path to save the filtered documents
output_file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/reviews3.json"

# Save filtered documents to a JSON file
with open(output_file_path, "w") as f:
    json.dump(filtered_documents, f)