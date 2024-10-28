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
import pandas as pd
import pprint

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from config import access_token, label_map, get_embedding_model, model_name

access_token = access_token
tokenizer = AutoTokenizer.from_pretrained(
    model_name, token=access_token, trust_remote_code=True
)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=access_token,
    device_map={"": 0},
    quantization_config=bnb_config,
    torch_dtype="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

prompt_in_chat_format = """
You are a helpful and knowledgeable sales agent assisting a customer. 
Please provide a brief response based solely on the following context, 
keeping the tone friendly and professional.

Context:
{detailed_information}

Customer Question:
{question}

Answer the question directly, without mentioning "based on the provided context."
"""

# Comment this line when you dont have the vector database
vector_database = FAISS.load_local(
    "/home/stud/abedinz1/localDisk/RAG/RAG/script/faiss_index",
    get_embedding_model(),
    allow_dangerous_deserialization=True,
)


def get_llm_response(question):

    messages = [
        {
            "role": "system",
            "content": f"""
            You are a helpful and knowledgeable sales agent assisting a customer. 
            Please provide a brief response, 
            keeping the tone friendly and professional.
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
        "/home/stud/abedinz1/localDisk/opinionconv-refactor/transformed_data_for_vector_database.json",
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
        for doc in ds
    ]

    db = FAISS.from_documents(
        raw_knowledge_base,
        get_embedding_model(),
        distance_strategy=DistanceStrategy.COSINE,
    )
    print("saving the data index")
    db.save_local("faiss_index")
    print("vectore db creatton done")


def get_vanilla_rag_response(question, llm):

    # create vector database
    if not os.path.exists("/home/stud/abedinz1/localDisk/RAG/RAG/script/faiss_index"):
        create_vector_database()

    relevant_doc = vector_database.similarity_search(query=question, k=1)

    pprint.pprint("Relevant Doc in vanilla:")
    pprint.pprint(relevant_doc)

    relevant_doc = relevant_doc[0]
    relevant_page_content = relevant_doc.page_content
    final_prompt = prompt_in_chat_format.format(
        question=question, detailed_information=relevant_doc
    )

    # Generate the answer using the large language model
    answer = llm(final_prompt)[0]["generated_text"]
    return answer, final_prompt


def get_our_rag_response(
    question, label, aspect, product_id, review_id, answer, bought_together=[]
):
    # Create vector database if not exists
    if not os.path.exists("/home/stud/abedinz1/localDisk/RAG/RAG/script/faiss_index"):
        create_vector_database()

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

    #print("relevant_docs after similarity_search search :",relevant_docs)

    # NOTE: This filter applies to all labels.
    # Thats why we dont have any seperate if condition for Opos1B_Opos1B2 label.
    filtered_docs = [
        doc for doc in relevant_docs if doc.metadata["reviewId"] != review_id
    ]
    # print("review_id: ",review_id)
    # print("2:", filtered_docs)
    if label == "Oneg1A_Opos2A" or label == "Opos1B_Opos2B" or label == "Opos1B_Oneg2B":
        filtered_docs = [
            doc for doc in relevant_docs if doc.metadata["aspect"] != aspect
        ]
    else:
        filtered_docs = [
            doc for doc in relevant_docs if doc.metadata["aspect"] == aspect
        ]

    # print("aspect: ",aspect)
    # print("3:", filtered_docs)
    if label == "Oneg1A_Opos1B":
        filtered_docs = [
            doc for doc in relevant_docs if doc.metadata["productId"] != product_id
        ]
    else:
        filtered_docs = [
            doc for doc in relevant_docs if doc.metadata["productId"] == product_id
        ]

    # print("productId: ",product_id)
    # print("filtered_docs after all filtering: ",filtered_docs)
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
            detailed_information = f"The answer should highlight positive attributes of the product's {aspect} and reference the same product with ID: {product_id}. Example: {relevant_doc.page_content}"
        elif label == "Oneg1A_Opos1A":
            detailed_information = f"The answer should focus on positive aspects of the product's {aspect} and reference the same product with ID: {product_id}. Example: {relevant_doc.page_content}"
        elif label == "Oneg1A_Opos1B":
            detailed_information = f"The answer should emphasize positive aspects of the {aspect}, referring to a different product from the one with ID: {product_id}. Example: {relevant_doc.page_content}"
        elif label == "Oneg1A_Opos2A":
            detailed_information = f"The answer should highlight positive aspects of a different aspect than {aspect}, focusing on the same product with ID: {product_id}. Example: {relevant_doc.page_content}"
        elif label == "Opos1B_Opos2B":
            detailed_information = f"The answer should mention positive aspects of a different attribute from {aspect}, referencing the same product with ID: {product_id}. Example: {relevant_doc.page_content}"
        elif label == "Opos1B_Opos1B2":
            detailed_information = f"The answer should provide positive information about {aspect}, focusing on the same product with ID: {product_id}. Example: {relevant_doc.page_content}"
        elif label == "Opos1B_Oneg2B":
            detailed_information = f"The answer should describe negative aspects of a different attribute than {aspect}, focusing on the same product with ID: {product_id}. Example: {relevant_doc.page_content}"

        messages = [
            {
                "role": "system",
                "content": f"""
                You are a helpful and knowledgeable sales agent assisting a customer. 
                Please provide a brief response based solely on the following context, 
                keeping the tone friendly and professional.

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

if __name__ == "__main__":
    # Load pickled data
    with open(
        "/home/stud/abedinz1/localDisk/RAG/RAG/data/question_answer_pairs.pkl", "rb"
    ) as f:
        blocks_neg_100 = pickle.load(f)

    counter = 0

    # Writing to CSV file
    with open(
        "output_file_path.csv", "w", newline="", encoding="utf-8"
    ) as output_file_path:
        fieldnames = [
            "query",
            "opinion_conv_response",
            "llm_response",
            #"vanilla_rag_response",
            #"vanilla_rag_prompt",
            "our_rag_response",
            #"our_rag_prompt",
            "label",
            #"our_relevant_doc",
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

            # Save llm response
            print("save llm response")
            llm_response = get_llm_response(question)
            print("\n\n")
            # Save vanilla rag response
            # print("save vanilla rag response")
            # vanilla_rag_response, vanilla_rag_prompt = get_vanilla_rag_response(
            #     question, get_reader_model()
            # )
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
                bought_together=[],
            )

            # Write in csv
            writer.writerow(
                {
                    "query": question,
                    "opinion_conv_response": opinion_conv_response,
                    "llm_response": llm_response,
                    # "vanilla_rag_response": vanilla_rag_response,
                    # "vanilla_rag_prompt": vanilla_rag_prompt,
                    "our_rag_response": our_rag_response,
                    #"our_rag_prompt": our_rag_prompt,
                    "label": label,
                    # "our_relevant_doc": relevant_doc,
                }
            )

            # counter+=1
            # if counter>3:
            #     break
