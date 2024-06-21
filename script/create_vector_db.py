import pandas as pd
from typing import List
from datasets import Dataset
import locale
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import pandas as pd
from typing import List, Tuple
from datasets import Dataset
import locale
import os
import json
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gzip
from transformers import AutoTokenizer
from config import (
    embedding_model,
    tokenizer
    )

def create_vector_db(docs_processed):
    """
    Create a FAISS vector database from processed documents and save the index locally.

    Args:
        docs_processed (list): A list of processed documents to be indexed.

    Returns:
        FAISS: The FAISS database created from the documents.
    """
    db = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    print("saving the data index")
    db.save_local("faiss_index")
    return db


prompt_in_chat_format = """
Answer the question only based on the following context:
Give a short answer and don't mention "based on the provided context"
-----


Context: {context}

---

Answer the question based on the above context: 
Question: {question}
"""


# Define a function to load JSON data from a file
def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# Define a function to save JSON data to a file
def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

# Function to process questions and answers using RAG


def load_json(file_path: str) -> List[dict]:
    """Load data from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    else:
        print(f"File '{file_path}' not found.")
        return []


### load the data
def parse(path):
    data = []
    with gzip.open(path) as f:
        for l in f:
            data.append(json.loads(l.strip()))
        return(data)

def create_huggingface_dataset(data: List[dict]) -> Dataset:
    """Create a Hugging Face dataset from JSON data."""
    review_texts = [item["reviewText"] for item in data]
    product_ids = [item["asin"] for item in data]
    reviewer_ids = [item["reviewerID"] for item in data]
    ds = Dataset.from_dict({"reviewText": review_texts, "asin": product_ids, "reviewerID": reviewer_ids})
    return ds

def write_reviews_to_text_file(reviews: List[str], file_path: str) -> None:
    """Write reviews to a text file."""
    with open(file_path, "w") as f:
        for review in reviews:
            f.write(review + "\n")

def apply_punctuation_model_to_text_file(input_file: str, output_file: str) -> None:
    """Apply punctuation model to a text file."""
    # Here you can implement your punctuation model to add punctuations
    # For simplicity, let's just add a period at the end of each line
    with open(input_file, "r") as f:
        with open(output_file, "w") as f_out:
            for line in f:
                f_out.write(line.strip() + ".\n")


def save_reviews_to_file(reviews: List[str], file_path: str) -> None:
    """Save reviews to a text file."""
    with open(file_path, "w") as f:
        for review in reviews:
            f.write(review + " #@#@#\n")

def format_reviews(input_file: str, output_file: str) -> None:
    """Separate reviews from the input file and write them to the output file."""
    # Read contents of the input file
    with open(input_file, 'r') as f:
        content = f.read()

    # Split the content based on the delimiter '#@#@#'
    reviews = content.replace('#@#@#,','\n').replace('#@#@#','\n').split("\n")[:-1]


    # Open the output file and write each review to a separate line
    with open(output_file, 'w') as f:
        for review in reviews:
            # Remove leading and trailing whitespaces
            review = review.strip()
            # If the review does not end with a full stop, add it
            if review and review[-1] != '.':
                review += '.'
            # Write the review to the output file
            f.write(review + '\n')



def create_dataset(data: List[dict]) -> Dataset:
    """Create a Hugging Face dataset from JSON data with punctuated reviews."""

    review_texts = [item.get('reviewText', " ")  for item in data]
    product_ids = [item["asin"] for item in data]
    reviewer_ids = [item["reviewerID"] for item in data]
    ds = Dataset.from_dict({"reviewText": review_texts, "asin": product_ids, "reviewerID": reviewer_ids})
    return ds

def preprocess_documents(ds: Dataset):
    """Preprocess documents for Langchain.
		So we iterate over every element of the dataset
		and make it langchain documents
    """
    raw_knowledge_base = [
        LangchainDocument(page_content=doc["reviewText"], metadata={"productId": doc["asin"], "reviewerID": doc["reviewerID"]})
        for doc in ds
    ]
    return raw_knowledge_base


def chunk_documents(raw_docs):

    #TODO: Test the AutoTokenizer change

    """Split documents using Langchain's RecursiveCharacterTextSplitter."""
    markdown_separators = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    #keep chunk size bigger than the longest review.
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=2000,
        chunk_overlap=100,
        add_start_index=True,
        strip_whitespace=True,
        separators=markdown_separators,
    )

    docs_processed = []

    for doc in raw_docs:
        docs_processed += text_splitter.split_documents([doc])
    return docs_processed


def main():
    pd.set_option("display.max_colwidth", None)
    locale.getpreferredencoding = lambda: "UTF-8"

    # File path to your JSON file
    file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/data/filtered_reviews.json"
    #file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/Cell_Phones_and_Accessories_5.json.gz"


    # Load data from JSON file
    data = load_json(file_path)
    #data = parse(file_path)

    # Create dataset
    ds = create_dataset(data)
    
    # Preprocess documents for Langchain
    raw_docs = preprocess_documents(ds)
    
    # Chunking
    docs_processed = chunk_documents(raw_docs)

    # Create Vector Database
    create_vector_db(docs_processed)

if __name__ == "__main__":
    main()



