import json
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores.utils import DistanceStrategy
from datasets import Dataset
import os
import csv
import pickle
import pandas as pd

from config import get_embedding_model, get_reader_model, conversation_mapping


prompt_in_chat_format = """
Answer the question only based on the following context:
Give a short answer and don't mention "based on the provided context"

Persona: You are a sales agent having a conversation with a customer
-----


detailed_information/context: {detailed_information}

---

Answer the question based on the above detailed_information/context and persona: 
Question: {question}
"""

llm = get_reader_model()

def get_llm_response(question, label="", aspect="", product_id=""):

    if label == "Qpos1A_Apos1A":
        detailed_information = f"The answer should have positive polarity about aspect: {aspect} and about same product with product id: {product_id}"
    elif label == "Oneg1A_Opos1A":
        detailed_information = f"The answer should have  positive polarity about aspect: {aspect} and about same product with product id: {product_id}"
    elif label == "Oneg1A_Opos1B":
        detailed_information = f"The answer should have  positive polarity about aspect: {aspect} but with different from product with product id: {product_id}"

    final_prompt = prompt_in_chat_format.format(
        question=question, detailed_information=detailed_information
    )

    # Generate the answer using the large language model
    answer = llm(final_prompt)[0]["generated_text"]
    return answer


def create_vector_database():
    # Load data using hugginface dataset
    # Load pickled data instead of JSON files
    with open("/home/stud/abedinz1/localDisk/RAG/RAG/data/question_answer_pairs.pkl", 'rb') as f:
        blocks_neg_100 = pickle.load(f)    

    # Check the structure of blocks_neg_100
    print(type(blocks_neg_100))
    print(blocks_neg_100)
    ds = Dataset.from_list(blocks_neg_100)


    # """Preprocess documents for Langchain."""
    raw_knowledge_base = [
        LangchainDocument(
            page_content=doc["sentence"],
            metadata={
                "productId": doc["asin"],
                "aspect": doc["aspect"],
                "polarity": doc["polarty"],
            },
        )
        for doc in ds
    ]

    db = FAISS.from_documents(
        raw_knowledge_base,
        get_embedding_model(),
        distance_strategy=DistanceStrategy.COSINE,
    )
    print("saving the data index")
    db.save_local("faiss_index")


def get_vanilla_rag_response(question):

    # create vector database
    if not os.path.exists("index path"):
        create_vector_database()

    vector_database = FAISS.load_local(
        "faiss_index", get_embedding_model(), allow_dangerous_deserialization=True
    )

    relevant_doc = vector_database.similarity_search(query=question, k=1)
    relevant_doc = relevant_docs[0]
    relevant_page_content = relevant_doc.page_content
    final_prompt = prompt_in_chat_format.format(
        question=question, detailed_information=relevant_doc
    )

    # Generate the answer using the large language model
    answer = get_reader_model(final_prompt)[0]["generated_text"]
    return answer


def get_our_rag_response(question, label, aspect, product_id):

    # create vector database
    if not os.path.exists("index path"):
        create_vector_database()

    vector_database = FAISS.load_local(
        "faiss_index", get_embedding_model(), allow_dangerous_deserialization=True
    )

    if label == "Qpos1A_Apos1A":
        filter_dict = {
            "productId": product_id,
            "aspect": aspect,
            "polarity": "positive",
        }
    elif label == "Oneg1A_Opos1A":
        filter_dict = {
            "productId": product_id,
            "aspect": aspect,
            "polarity": "positive",
        }
    elif label == "Oneg1A_Opos1B":
        filter_dict = {
            "productId": {"$ne": product_id},
            "aspect": aspect,
            "polarity": "positive",
        }

    relevant_doc = vector_database.similarity_search(
        query=question, filter=filter_dict, k=1
    )
    relevant_doc = relevant_doc[0]
    relevant_page_content = relevant_doc.page_content
    final_prompt = prompt_in_chat_format.format(
        question=question, detailed_information=relevant_doc
    )

    # Generate the answer using the large language model
    answer = get_reader_model(final_prompt)[0]["generated_text"]
    return answer

# Load pickled data instead of JSON files
with open("/home/stud/abedinz1/localDisk/RAG/RAG/data/question_answer_pairs.pkl", 'rb') as f:
    blocks_neg_100 = pickle.load(f)

with open("output_file_path", "w", newline="", encoding="utf-8") as output_file_path:
    fieldnames = [
        "query",
        "opinion_conv_response",
        "llm_response",
        "vanilla_rag_response",
        "our_rag_response",
    ]

    writer = csv.DictWriter(output_file_path, fieldnames=fieldnames)
    writer.writeheader()

    for item in blocks_neg_100:

        # extract information from object
        question = item["question"]
        product_id = item["product_id"]
        label = item["label"]
        aspect = item["aspect"]
        answer = item["answer"]

        # save OpinionConv Response
        opinion_conv_response = answer

        # save llm response
        llm_response = get_llm_response(question, label, aspect, product_id)

        # save vanilla rag response
        vanilla_rag_response = get_vanilla_rag_response(question)

        # save our rag response
        our_rag_response = get_our_rag_response(question, label, aspect, product_id)

        # write in csv
        writer.writerow(
            {
                "query": question,
                "opinion_conv_response": opinion_conv_response,
                "llm_response": llm_response,
                "vanilla_rag_response": vanilla_rag_response,
                "our_rag_response": our_rag_response,
            }
        )
