from tqdm.notebook import tqdm
from tqdm.gui import tqdm
import pandas as pd
from typing import List, Tuple
from datasets import Dataset
import locale
import os
import json
import pdb
from vector_database import vector_data_base_createion, retrieval_top_k, answer_with_rag, READER_LLM, RAG_PROMPT_TEMPLATE
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

import tempfile


def load_json(file_path: str) -> List[dict]:
    """Load data from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    else:
        print(f"File '{file_path}' not found.")
        return []


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



def create_huggingface_dataset_with_punctuation(data: List[dict]) -> Dataset:
    """Create a Hugging Face dataset from JSON data with punctuated reviews."""
    review_texts = [item["reviewText"] for item in data]

    review_texts_file = "/home/s28zabed/RAG/reviews_text_file.txt"
    punctuated_review_texts_file = "/home/s28zabed/RAG/punctuated_reviews.txt"
    review_texts_from_model = "/home/s28zabed/RAG/reviews_text_from_model.txt"
    
    # Save review texts to a text file, with f.write(review + "#@#@#\n")
    save_reviews_to_file(review_texts, review_texts_file)
    
    format_reviews(review_texts_from_model, punctuated_review_texts_file)

    
    # Read punctuated reviews from the text file
    punctuated_reviews = []
    with open(punctuated_review_texts_file, "r") as f:
        for line in f:
            punctuated_reviews.append(line.strip())
    
    #print(data)
    product_ids = [item["asin"] for item in data]
    reviewer_ids = [item["reviewerID"] for item in data]
    ds = Dataset.from_dict({"reviewText": punctuated_reviews, "asin": product_ids, "reviewerID": reviewer_ids})
    return ds

def preprocess_documents(ds: Dataset) -> List[LangchainDocument]:
    """Preprocess documents for Langchain."""
    raw_knowledge_base = [
        LangchainDocument(page_content=doc["reviewText"], metadata={"productId": doc["asin"], "reviewerID": doc["reviewerID"]})
        for doc in tqdm(ds)
    ]
    return raw_knowledge_base


def split_documents(raw_docs: List[LangchainDocument]) -> List[LangchainDocument]:
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
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
    #pdb.set_trace()

    # File path to your JSON file
    file_path = "/home/s28zabed/RAG/reviews2.json"

    # Load data from JSON file
    data = load_json(file_path)

    # Create a Hugging Face dataset
    ds = create_huggingface_dataset_with_punctuation(data)
    #print(ds)
    print("zain was hereeeee")

    # Preprocess documents for Langchain
    raw_docs = preprocess_documents(ds)
    print(raw_docs)
    
    #Split documents using Langchain's text splitter: Chunking
    docs_processed = split_documents(raw_docs)

    #Print the first processed document
    #print(docs_processed)
    user_query = "I have heard that the quality of the product is not that good"

    vector_database = vector_data_base_createion(docs_processed)
    retrieval_top_k(user_query, vector_database)
    print ("######zain####")

    question = "is this a good calendar?"

    answer, relevant_docs = answer_with_rag(
        question, READER_LLM, vector_database
    )




if __name__ == "__main__":
    main()


