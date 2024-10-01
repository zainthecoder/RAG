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

vector_database = FAISS.load_local(
        "/home/stud/abedinz1/localDisk/RAG/RAG/script/faiss_index", get_embedding_model(), allow_dangerous_deserialization=True
    )
def transform_data(data):
    transformed_data = []
    counter = 0
    for entry in data:
        aspects = entry.get("aspect", [])
        sentiments = entry.get("sentiment", [])
        
        # If there are multiple aspects and sentiments, create a new object for each pair
        if len(aspects) > 1 or len(sentiments) > 1:
            # Handle cases where the length of aspects and sentiments are unequal
            for i in range(max(len(aspects), len(sentiments))):
                new_entry = entry.copy()
                new_entry["aspect"] = [aspects[i]] if i < len(aspects) else []
                new_entry["sentiment"] = [sentiments[i]] if i < len(sentiments) else []
                transformed_data.append(new_entry)
        else:
            transformed_data.append(entry)
        counter = counter+1
        if counter>10:
            break
    return transformed_data

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

    print("Createing Vector Database")

    with open("/home/stud/abedinz1/localDisk/RAG/RAG/data/final_reviews_after_absa.json", 'r') as file:
        data = json.load(file)
    
    # Transform the data
    transformed_data = transform_data(data)

    ds = Dataset.from_list(transformed_data)


    # """Preprocess documents for Langchain."""
    raw_knowledge_base = [
        LangchainDocument(
            page_content=doc["text"],
            metadata={
                "productId": doc["asin"],
                "aspect": doc["aspect"][0] if doc["sentiment"] else "",
                "polarity": doc["sentiment"][0] if doc["sentiment"] else "",
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
    if not os.path.exists("/home/stud/abedinz1/localDisk/RAG/RAG/script/faiss_index"):
        create_vector_database()

    # vector_database = FAISS.load_local(
    #     "/home/stud/abedinz1/localDisk/RAG/RAG/script/faiss_index", get_embedding_model(), allow_dangerous_deserialization=True
    # )

    relevant_doc = vector_database.similarity_search(query=question, k=1)
    print("Relevant Doc in vanilla:")
    print(relevant_doc)
    relevant_doc = relevant_doc[0]
    relevant_page_content = relevant_doc.page_content
    final_prompt = prompt_in_chat_format.format(
        question=question, detailed_information=relevant_doc
    )

    # Generate the answer using the large language model
    answer = get_reader_model(final_prompt)[0]["generated_text"]
    return answer


def get_our_rag_response(question, label, aspect, product_id):

    # create vector database
    if not os.path.exists("/home/stud/abedinz1/localDisk/RAG/RAG/script/faiss_index"):
        create_vector_database()

    # vector_database = FAISS.load_local(
    #     "/home/stud/abedinz1/localDisk/RAG/RAG/script/faiss_index", get_embedding_model(), allow_dangerous_deserialization=True
    # )

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

if __name__ == '__main__':
    # Load pickled data
    with open("/home/stud/abedinz1/localDisk/RAG/RAG/data/question_answer_pairs.pkl", 'rb') as f:
        blocks_neg_100 = pickle.load(f)

    # Writing to CSV file
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
            # Extract information
            question = item["question"]
            product_id = item["product_id"]
            label = item["label"]
            aspect = item["aspect"]
            answer = item["answer"]

            # Save OpinionConv Response
            print("# save OpinionConv Response")
            opinion_conv_response = answer

            # Save llm response
            print("save llm response")
            llm_response = get_llm_response(question, label, aspect, product_id)

            # Save vanilla rag response
            print("save vanilla rag response")
            vanilla_rag_response = get_vanilla_rag_response(question)

            # Save our rag response
            print("save our rag response")
            our_rag_response = get_our_rag_response(question, label, aspect, product_id)

            # Write in csv
            writer.writerow({
                "query": question,
                "opinion_conv_response": opinion_conv_response,
                "llm_response": llm_response,
                "vanilla_rag_response": vanilla_rag_response,
                "our_rag_response": our_rag_response,
            })
