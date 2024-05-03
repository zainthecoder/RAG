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

vector_database = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)




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


# def search_neg_qa_pairs(data, rag_key, question_key):
#     for item in data:        
#         print(item)
#         if  item["key_question"] == question_key and item["key_answer"] == rag_key:
#             return item
#     print("we did not find , we will return, NOT PRESENT")
#     return {"answer": "Not present"}
# def search_neg_qa_pairs(data, rag_key, question_key):
#     lookup_dict = {}
#     for item in data:
#         key_pair = (item["key_question"], item["key_answer"])
#         lookup_dict[key_pair] = item
        
#     key_pair_to_find = (question_key, rag_key)
#     if key_pair_to_find in lookup_dict:
#         return lookup_dict[key_pair_to_find]
#     else:
#         print("Not present")
#         return {"answer": "Not present"}

import pandas

def search_neg_qa_pairs(data, rag_key, question_key):
    # Convert the list of dictionaries to a cuDF dataframe
    df = pandas.DataFrame(data)
    
    # Create a new column with the combined key pair
    df['key_pair'] = df['key_question'] + '_' + df['key_answer']
    
    # Search for the desired key pair
    result = df[df['key_pair'] == (question_key + '_' + rag_key)]
    
    if len(result) > 0:
        # Return the first matching row
        return result.iloc[0].to_dict()
    else:
        print("Not present")
        return {"answer": "Not present"}



# Function to process questions and answers using RAG
def process_questions_and_answers(input_file_path, output_file_path, filter):
    # Load JSON data from input file
    data = load_json(input_file_path)
    #vector_database = vector_data_base_createion(docs_processed)
    # Iterate over each question and answer pair
    for item in data:
        question = item["question"]
        product_id = item["key_question"].split('_')[0]
        question_key = item["key_question"]
        answer_key = item["key_answer"]


        answer, relevant_docs, final_prompt, rag_key = answer_with_rag(
            question, READER_LLM, vector_database, product_id, question_key, answer_key, filter
        )
        #print("$$$$$$$")
        #print("anser from rag: ", rag_key)
        #print("answer_key: ",answer_key)
        ##print("question_key: ", question_key)
        #print("$$$$$$$$")
        if rag_key == answer_key:
            print("rag key is same as anaswer key")
            search_pair= {"answer": "Not needed"}
        else:    
            print("The rag is different, now we will search")
            search_pair = search_neg_qa_pairs(data, rag_key, question_key)

        answer = answer
        item["answer_from_rag"] = answer
        item["final_prompt"] = final_prompt
        item["search_answer"] = search_pair["answer"]


        save_json_append([item], output_file_path)  # Append mode

    print(f"Processed data saved to {output_file_path}")



def answer_with_rag(
    question,
    llm,
    knowledge_index,
    product_id,
    question_key, 
    answer_key,
    filter,
    num_retrieved_docs: int = 1,
    num_docs_final: int = 1,
):
    #print("=> Retrieving documents...")
    #print("question:",question)
    #print("productId:",product_id)

    if filter=="1":
        #print("i am in filter retrieval")
        relevant_docs = knowledge_index.similarity_search(
            query=question,
            filter=dict(productId=product_id),
            k=1,
            fetch_k=1000
        )
    else:
        #print("i am in non filter retrieval")
        relevant_docs = knowledge_index.similarity_search(
            query=question,
            k=1
        )

    #print("zain relevant_docs:",relevant_docs)
    #relevant_docs, metadata = [(doc.page_content, doc.metadata) for doc in relevant_docs]  # keep only the text

    #relevant_docs = relevant_docs[:num_docs_final]
    #metadata = metadata[:num_docs_final]
    #print("rag_reviewer_id: ",metadata.reviewerId)
    #print("rag_product_id: ",metadata.productId)
    rag_key = " "
    if relevant_docs:
        relevant_doc = relevant_docs[0]
        relevant_page_content = relevant_doc.page_content
        metadata = relevant_doc.metadata
        rag_reviewer_id = metadata.get("reviewerID", None)
        rag_product_id = metadata.get("productId", None)
        #print("rag_reviewer_id: ",rag_reviewer_id)
        #print("rag_product_id: ",rag_product_id)
        rag_key = rag_product_id+"_"+rag_reviewer_id
        #print("rag_key: ",rag_key)
    else:
        print("no relevant_doc")


    final_prompt = prompt_in_chat_format.format(question=question, context=relevant_page_content)
    #final_prompt = prompt2.format(question=question, context=relevant_docs)

    #print("final promp:", final_prompt)
    #print("relevant_docs:",relevant_docs)
    ## Redact an answer
    #print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]
    #pprint.pprint(answer)
    #print("#############")
    #print("\n\n\n\n\n")

    # Check if retrieved document matches the answer
    # _, _, neg_qa_pairs = read_json(neg_qa_pairs_file)
    # matching_pair = search_neg_qa_pairs(product_id, review_id, neg_qa_pairs)

    # if matching_pair:
    #     return matching_pair["question"], matching_pair["answer"], answer
    # else:
    #     return "No matching pair found in neg_qa_pairs.json", None, answer

    return answer, relevant_docs, final_prompt, rag_key




# # Process questions and answers using RAG
def main_run():
    args = sys.argv[1:]
    output_file_name = args[0]
    filter = args[1]
    print("filter: ",filter)
    output_file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/" + output_file_name
    print(output_file_path)

    #input_file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/combined_qa_pairs.json"
    input_file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/data/filtered_qa_pairs.json"
    process_questions_and_answers(input_file_path, output_file_path,filter)



if __name__ == "__main__":
    main_run()