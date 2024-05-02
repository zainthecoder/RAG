from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import locale
import json
import sys

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS

import pdb
from transformers import pipeline
import torch
import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import (
    embedding_model,
    READER_LLM
    )

#input_file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/qa_pairs.json"
input_file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/neg_qa_pairs.json"
#output_file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/output_rag.json"
output_file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/neg_output_rag.json"



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

# Function to save data to a JSON file
def save_json_append(data, file_path):
    with open(file_path, 'a') as f:  # Open file in append mode
        json.dump(data, f)
        f.write('\n')  # Write newline character to separate JSON objects


def search_neg_qa_pairs(product_id, review_id, neg_qa_pairs):
    for pair in neg_qa_pairs:
        key_question_product_id, key_question_review_id, _, _ = pair["key_question"].split("_")
        key_answer_product_id, key_answer_review_id, _, _ = pair["key_answer"].split("_")

        if key_question_product_id == product_id and key_question_review_id == review_id:
            return pair
        elif key_answer_product_id == product_id and key_answer_review_id == review_id:
            return pair

    return None


# Function to process questions and answers using RAG
def process_questions_and_answers(input_file_path, output_file_path):
    # Load JSON data from input file
    print("yahiiooooo")
    data = load_json(input_file_path)
    #vector_database = vector_data_base_createion(docs_processed)
    vector_database = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
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

        save_json_append([item], output_file_path)  # Append mode

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
    relevant_docs = knowledge_index.similarity_search(
        query=question,
        #filter=dict(productId=product_id),
        k=1
        #fetch_k=1000
        )
    
    print("zain relevant_docs:",relevant_docs)
    #relevant_docs, metadata = [(doc.page_content, doc.metadata) for doc in relevant_docs]  # keep only the text

    #relevant_docs = relevant_docs[:num_docs_final]
    #metadata = metadata[:num_docs_final]
    #print("rag_reviewer_id: ",metadata.reviewerId)
    #print("rag_product_id: ",metadata.productId)


    relevant_doc = relevant_docs[0]
    relevant_page_content = relevant_doc.page_content
    metadata = relevant_doc.metadata
    rag_reviewer_id = metadata.get("reviewerID", None)
    rag_product_id = metadata.get("productId", None)
    print("rag_reviewer_id: ",rag_reviewer_id)
    print("rag_product_id: ",rag_product_id)


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


    # # Check if retrieved document matches the answer
    # _, _, neg_qa_pairs = read_json(neg_qa_pairs_file)
    # matching_pair = search_neg_qa_pairs(product_id, review_id, neg_qa_pairs)

    # if matching_pair:
    #     return matching_pair["question"], matching_pair["answer"], answer
    # else:
    #     return "No matching pair found in neg_qa_pairs.json", None, answer

    return answer, relevant_docs, final_prompt




# # Process questions and answers using RAG
def main():
    args = sys.argv[1:]
    output_file_name = args[0]
    output_file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/" + output_file_name
    print(output_file_path)
    process_questions_and_answers(input_file_path, output_file_path)



if __name__ == "__main__":
    main()