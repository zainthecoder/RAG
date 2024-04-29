from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import locale

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS

import pdb
from transformers import pipeline
import torch
import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
#READER_MODEL_NAME = "gpt2-medium"



#hello

# EMBEDDING_MODEL_NAME = "thenlper/gte-small"

# embedding_model = HuggingFaceEmbeddings(
#     model_name=EMBEDDING_MODEL_NAME,
#     multi_process=True,
#     model_kwargs={"device": "cuda"},
#     encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
# )

def vector_data_base_createion(docs_processed):
    print(docs_processed)
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    return KNOWLEDGE_VECTOR_DATABASE

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )
# model = AutoModelForCausalLM.from_pretrained(
#     READER_MODEL_NAME, quantization_config=bnb_config
# )
# tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

#READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
# READER_LLM = pipeline(
#         model=model,
#         tokenizer=tokenizer,
#         task="text-generation",
#         do_sample=True,
#         temperature=0.2,
#         repetition_penalty=1.1,
#         return_full_text=False,
#         max_new_tokens=500,
# )



import json

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
def process_questions_and_answers(input_file_path, output_file_path, docs_processed):
    # Load JSON data from input file
    print("yahiiooooo")
    data = load_json(input_file_path)
    vector_database = vector_data_base_createion(docs_processed)
    # Iterate over each question and answer pair
    for item in data:
        question = item["question"]
        product_id = item["key_question"].split('_')[0]

        answer, relevant_docs, final_prompt = answer_with_rag(
            question, READER_LLM, vector_database, product_id
        )


        answer = answer
        item["answer_from_rag"] = answer
        item["final_prompt"] = final_prompt

    # Save the updated data to a new JSON file
    save_json(data, output_file_path)
    print(f"Processed data saved to {output_file_path}")


def answer_with_rag(
    question,
    llm,
    knowledge_index,
    product_id,
    num_retrieved_docs: int = 1,
    num_docs_final: int = 1,
):
    # Gather documents with retriever
    print("=> Retrieving documents...")
    # relevant_docs = knowledge_index.similarity_search(
    #     query=question, k=num_retrieved_docs
    # )
    print("question:",question)
    print("productId:",product_id)
    relevant_docs = knowledge_index.similarity_search_with_score(
        query=question,
        filter=dict(productId=product_id),
        k=1
        )
    # relevant_docs = knowledge_index.similarity_search(
    #     query=question, k=1
    # )
    print("zain relevant_docs:",relevant_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # keep only the text

    relevant_docs = relevant_docs[:num_docs_final]


    final_prompt = prompt_in_chat_format.format(question=question, context=relevant_docs)
    #final_prompt = prompt2.format(question=question, context=relevant_docs)

    print("final promp:", final_prompt)
    print("relevant_docs:",relevant_docs)
    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]
    pprint.pprint(answer)
    print("#############")
    print("\n\n\n\n\n")

    return answer, relevant_docs, final_prompt


    import pandas as pd
from typing import List, Tuple
from datasets import Dataset
import locale
import os
import json
import pdb
#from vector_database import vector_data_base_createion, retrieval_top_k, answer_with_rag, READER_LLM, RAG_PROMPT_TEMPLATE
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

import tempfile
import gzip

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



def create_huggingface_dataset_with_punctuation(data: List[dict]) -> Dataset:
    """Create a Hugging Face dataset from JSON data with punctuated reviews."""

    review_texts = [item.get('reviewText', " ") for item in data]
    # review_texts = []
    # for item in data:
    #     if item.get('reviewText'):
    #         review_texts.append(item['reviewText'])
    #     else:
    #         print("I guess review is not there")
    #         print(item.get('reviewText'))
    #         review_texts.append(" ")


    #review_texts_file = "/content/drive/MyDrive/RAG/reviews_text_file.txt"
    ##punctuated_review_texts_file = "/content/drive/MyDrive/RAG/punctuated_reviews.txt"
    #review_texts_from_model = "/content/drive/MyDrive/RAG/reviews_text_from_model.txt"

    # Save review texts to a text file, with f.write(review + "#@#@#\n")
    #save_reviews_to_file(review_texts, review_texts_file)

    #format_reviews(review_texts_from_model, punctuated_review_texts_file)


    #Read punctuated reviews from the text file
    #punctuated_reviews = []
    #with open(punctuated_review_texts_file, "r") as f:
    ##    for line in f:
    #        punctuated_reviews.append(line.strip())

    #print(data)
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


def split_documents(raw_docs):
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
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
    #file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/reviews3.json"
    #file_path = "/content/drive/MyDrive/RAG/1000_reviews.json"
    file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/Cell_Phones_and_Accessories_5.json.gz"


    # Load data from JSON file
    #data = load_json(file_path)
    data = parse(file_path)
    print(data[0])
    # Create a Hugging Face dataset
    ds = create_huggingface_dataset_with_punctuation(data)
    print(ds[0])
    # print(ds[-1])

   

    # Preprocess documents for Langchain
    raw_docs = preprocess_documents(ds)
    #pprint.pprint(raw_docs)


    print("Document after chunking")
    #Split documents using Langchain's text splitter: Chunking
    docs_processed = split_documents(raw_docs)
    pprint.pprint(docs_processed[0])

    # print("zain is here againnn")



    # Define input and output file paths
    # now we need to create qa_pairs for the entire dataset
    input_file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/qa_pairs_new.json"
    output_file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/rag_output.json"

    # # Process questions and answers using RAG
    # process_questions_and_answers(input_file_path, output_file_path,docs_processed)



if __name__ == "__main__":
    main()



