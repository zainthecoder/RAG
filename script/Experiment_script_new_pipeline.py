import json
import torch

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores.utils import DistanceStrategy
from datasets import Dataset
from collections import OrderedDict

import os
import csv
import pickle
import pprint

from config import label_map, get_tokenizer, get_embedding_model, get_reader_model
from datasets import load_dataset

base_dir = "/home/s28zabed"

def get_llm_response(question, product_name, model, tokenizer):

    messages = [
        {
            "role": "system",
            "content": f"""
            You are a helpful and knowledgeable sales agent assisting a customer. 
            Please provide a brief response to the following question related to this product: {product_name} 
            """,
        },
        {
            "role": "user",
            "content": f"""
            Customer Question:
            {question}
            Answer the question directly"
        """,
        },
    ]

    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
        "cuda"
    )

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    generated_data = tokenizer.batch_decode(generated_ids)[0]

    print("\nResonse")
    pprint.pprint(generated_data)
    return generated_data


def create_vector_database():
    # Load data using hugginface dataset

    print("Creating Vector Database")

    with open(
        base_dir+"/opinionconv-refactor/transformed_data_for_vector_database.json",
        "r",
    ) as file:
        data = json.load(file)

    # Transform the data
    # transformed_data = transform_data(data)

    ds = Dataset.from_list(data)
    document_count = len(ds)
    print(f"Number of documents in the dataset: {document_count}")
    c = 0
    for doc in ds:
        print("\n")
        pprint.pprint(doc)
        print(f'{doc["asin"]}_{doc["user_id"]}')
        c += 1
        if c > 5:
            break

    # """Preprocess documents for Langchain."""
    
    for doc in ds:
        counter =0
        raw_knowledge_base = [
            LangchainDocument(
                page_content=doc["sentence"],
                metadata={
                    "productId": doc["asin"],
                    "aspect": doc["aspect"],
                    "polarity": doc["sentiment"],
                    "reviewId": f'{doc["asin"]}_{doc["user_id"]}_{doc["unique_key"]}',
                },
            )
        ]
        #TODO: remove this check
        # counter+=1
        # if counter>5:
        #     break

    db = FAISS.from_documents(
        raw_knowledge_base,
        get_embedding_model(),
        distance_strategy=DistanceStrategy.COSINE,
    )
    print("saving the data index")
    db.save_local("faiss_index")
    print("vectore db creatton done")


def get_vanilla_rag_response(question, product_name, model, tokenizer, vector_database):

    relevant_doc = vector_database.similarity_search(query=question, k=10)

    pprint.pprint("Relevant Doc in vanilla:")
    pprint.pprint(relevant_doc)

    filtered_docs = [
        doc for doc in relevant_doc if doc.metadata["reviewId"] != review_id
    ]

    relevant_doc = filtered_docs[0]

    detailed_information = ""

    if label == "Qpos1A_Apos1A":
        detailed_information = f"The answer should focus on following product: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"
    elif label == "Oneg1A_Opos1A":
        detailed_information = f"The answer should focus on following product: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"
    elif label == "Oneg1A_Opos1B":
        detailed_information = f"The answer should focus on a different product from the one with name: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"
    elif label == "Oneg1A_Opos2A":
        detailed_information = f"The answer should focus on the following product: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"
    elif label == "Opos1B_Opos2B":
        detailed_information = f"The answer should focus on following product: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"
    elif label == "Opos1B_Opos1B2":
        detailed_information = f"The answer should focus on a different product from the one with name: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"
    elif label == "Opos1B_Oneg2B":
        detailed_information = f"The answer should focus on following product: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"

    messages = [
        {
            "role": "system",
            "content": f"""
            You are a helpful and knowledgeable sales agent assisting a customer. 
            Please provide a brief response based solely on the following context

            Context:{detailed_information} 
            """,
        },
        {
            "role": "user",
            "content": f"""
            Customer Question:
            {question}
            Answer the question directly, without mentioning "based on the provided context."
        """
        }
    ]

    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
            "cuda"
        )

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    generated_data = tokenizer.batch_decode(generated_ids)[0]

    print("\nResonse from our rag response")
    pprint.pprint(generated_data)
    answer = generated_data

    return answer

def get_our_rag_response(
    question, label, aspect, product_id, review_id, answer, product_name, model, tokenizer, vector_database
):
    # Define default filter and search settings
    base_filter = {
        "polarity": "Positive",
    }

    if label == "Opos1B_Oneg2B":
        base_filter["polarity"] = "Negative"

    # Perform similarity search
    relevant_docs = vector_database.similarity_search(
        query=question,
        filter=base_filter,
        k=500,  # Number of results to return
        fetch_k=96206,  # Number of results to fetch before filtering
    )

    # NOTE: This filter applies to all labels.
    # Thats why we dont have any seperate if condition for Opos1B_Opos1B2 label.
    filtered_docs = [
        doc for doc in relevant_docs if doc.metadata["reviewId"] != review_id
    ]

    if label == "Oneg1A_Opos2A" or label == "Opos1B_Opos2B" or label == "Opos1B_Oneg2B":
        filtered_docs = [
            doc for doc in relevant_docs if doc.metadata["aspect"] != aspect
        ]
    else:
        filtered_docs = [
            doc for doc in relevant_docs if doc.metadata["aspect"] == aspect
        ]
   
    if label == "Oneg1A_Opos1B":
        filtered_docs = [
            doc for doc in relevant_docs if doc.metadata["productId"] != product_id
        ]
    else:
        filtered_docs = [
            doc for doc in relevant_docs if doc.metadata["productId"] == product_id
        ]

    # Generate the response using LLM
    answer = ""
    final_prompt = ""
    relevant_doc = ""
    messages = []
    generated_data = ""
    if filtered_docs:
        relevant_doc = filtered_docs[0]

        detailed_information = ""

        if label == "Qpos1A_Apos1A":
            detailed_information = f"The answer should focus on positive aspects of the {aspect} of the following product: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"
        elif label == "Oneg1A_Opos1A":
            detailed_information = f"The answer should focus on positive aspects of the {aspect} of the following product: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"
        elif label == "Oneg1A_Opos1B":
            detailed_information = f"The answer should focus on positive aspects of the {aspect}, referring to a different product from the one with name: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"
        elif label == "Oneg1A_Opos2A":
            detailed_information = f"The answer should mention positive aspects of a different attribute from {aspect}, referencing the same product with name: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"
        elif label == "Opos1B_Opos2B":
            detailed_information = f"The answer should mention positive aspects of a different attribute from {aspect}, referencing the same product with name: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"
        elif label == "Opos1B_Opos1B2":
            detailed_information = f"The answer should focus on positive aspects of the {aspect}, referring to a different product from the one with name: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"
        elif label == "Opos1B_Oneg2B":
            detailed_information = f"The answer should describe negative aspects of a different attribute than {aspect}, focusing on the same product with name: {product_name}. Answer the question using the following review for this product: {relevant_doc.page_content}"

        messages = [
            {
                "role": "system",
                "content": f"""
                You are a helpful and knowledgeable sales agent assisting a customer. 
                Please provide a brief response based solely on the following context

                Context:{detailed_information} 
                """,
            },
            {
                "role": "user",
                "content": f"""
                Customer Question:
                {question}
                Answer the question directly, without mentioning "based on the provided context."
            """
            }
        ]

        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
            "cuda"
        )

        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        generated_data = tokenizer.batch_decode(generated_ids)[0]

        print("\nResonse from our rag response")
        pprint.pprint(generated_data)
        answer = generated_data

        return answer
    return answer




# Load dataset once and pass it as an argument to relevant functions
def load_metadata_dataset():
    df_metaData_raw_cellPhones = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Cell_Phones_and_Accessories",
        split="full", #TODO: change this to full
        trust_remote_code=True,
    )

    return df_metaData_raw_cellPhones

if __name__ == "__main__":
    
    print("Hi")
    
    # Load metadata dataset once here
    df_metaData_raw_cellPhones = load_metadata_dataset()
    parent_asin_to_title = {row['parent_asin']: row['title'] for row in df_metaData_raw_cellPhones}
    
    print("zain")
    # create vector database
    if not os.path.exists(base_dir+"/RAG/script/faiss_index"):
        create_vector_database()

    print("zain1")
    vector_database = FAISS.load_local(
        base_dir+"/RAG/script/faiss_index",
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )

    print("zain2")

    # Load pickled data
    with open(
        base_dir+"/RAG/data/neg_question_answer_pairs.pkl", "rb"
    ) as f:
        blocks_neg_100 = pickle.load(f)
    
    # Load pickled data
    #TODO: change it to pos
    with open(
        base_dir+"/RAG/data/pos_question_answer_pairs.pkl", "rb"
    ) as f:
        blocks_pos_100 = pickle.load(f)
    
    blocks_neg_100.extend(blocks_pos_100)

    counter = 0
    print("wassup")
    # Writing to CSV file
    with open(
        "output_file_path.csv", "w", newline="", encoding="utf-8"
    ) as output_file_path:
        fieldnames = [
            "query",
            "opinion_conv_response",
            "llm_response",
            "vanilla_rag_response",
            "our_rag_response",
            "label",
        ]
        writer = csv.DictWriter(output_file_path, fieldnames=fieldnames)
        writer.writeheader()

        for item in blocks_neg_100:
        #for i in range(0, len(blocks_neg_100), 2000):
            #item = blocks_neg_100[i]
            print("\n\n")
            # Extract information
            question = item["question"]
            product_id = item["product_id"]
            label = item["label"]
            aspect = item["aspect"]
            answer = item["answer"]
            review_id = item["review_id"]
            # bought_together = item["bought_together"]
            # print("Label: ", label)
            # label = label_map[label]
            # print("Label: ", label)
            # Save OpinionConv Response
            # print("# save OpinionConv Response")
            opinion_conv_response = answer

            
            
            print("Label: ",label)
            label = label_map[label]
            print("Label: ",label)

            #TODO: Undo this
            #product_name = parent_asin_to_title[product_id]
            product_name = "test"

            # Save llm response
            print("save llm response")
            llm_response = get_llm_response(question, product_name, get_reader_model(), get_tokenizer())
            
            print("\n\n")
            #Save vanilla rag response
            print("save vanilla rag response")
            vanilla_rag_response  = get_vanilla_rag_response(
                question, product_name, get_reader_model(), get_tokenizer(), vector_database
            )
            
            print("\n\n")
            #Save our rag response
            print("save our rag response")
            our_rag_response = get_our_rag_response(
                question,
                label,
                aspect,
                product_id,
                review_id,
                answer,
                product_name,
                get_reader_model(),
                get_tokenizer(),
                vector_database
            )

            # Write in csv
            writer.writerow(
                {
                    "query": question,
                    "opinion_conv_response": opinion_conv_response,
                    "llm_response": llm_response,
                    "vanilla_rag_response": vanilla_rag_response,
                    "our_rag_response": our_rag_response,
                    "label": label,
                }
            )

            # counter+=1
            # if counter>3:
            #     break
