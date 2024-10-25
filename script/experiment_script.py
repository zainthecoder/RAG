import json
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

from config import get_embedding_model, get_reader_model, conversation_mapping, label_map


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


def get_llm_response(question, llm, label="", aspect="", product_id=""):
    
    detailed_information = ""
  
    if label == "Qpos1A_Apos1A":
        detailed_information = f"The answer should highlight positive attributes of the product's {aspect} and reference the same product with ID: {product_id}."
    elif label == "Oneg1A_Opos1A":
        detailed_information = f"The answer should focus on positive aspects of the product's {aspect} and reference the same product with ID: {product_id}."
    elif label == "Oneg1A_Opos1B":
        detailed_information = f"The answer should emphasize positive aspects of the {aspect}, referring to a different product from the one with ID: {product_id}."
    elif label == "Oneg1A_Opos2A":
        detailed_information = f"The answer should highlight positive aspects of a different aspect than {aspect}, focusing on the same product with ID: {product_id}."
    elif label == "Opos1B_Opos2B":
        detailed_information = f"The answer should mention positive aspects of a different attribute from {aspect}, referencing the same product with ID: {product_id}."
    elif label == "Opos1B_Opos1B2":
        detailed_information = f"The answer should provide positive information about {aspect}, focusing on the same product with ID: {product_id}."
    elif label == "Opos1B_Oneg2B":
        detailed_information = f"The answer should describe negative aspects of a different attribute than {aspect}, focusing on the same product with ID: {product_id}."



    final_prompt = prompt_in_chat_format.format(
        question=question, detailed_information=detailed_information
    )

    pprint.pprint(final_prompt)
    # Generate the answer using the large language model
    answer = llm(final_prompt)[0]["generated_text"]
    pprint.pprint(answer)

    return answer


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
    question, label, aspect, product_id, review_id, answer, llm, bought_together=[]
):
    # Create vector database if not exists
    if not os.path.exists("/home/stud/abedinz1/localDisk/RAG/RAG/script/faiss_index"):
        create_vector_database()

    # Define default filter and search settings
    base_filter = {
        "productId": product_id,
        "aspect": aspect,
        "polarity": "Positive",
    }

    # # Modify filters based on label
    # if label == "Qpos1A_Apos1A" or label == "Oneg1A_Opos1A":
    #     print(f"label: {label}")
    #     # Same as base_filter

    # elif label == "Oneg1A_Opos1B":
    #     #print(f"label: {label}")
    #     # base_filter['productId'] = {"$in": bought_together}

    # elif label == "Oneg1A_Opos2A" or label == "Opos1B_Opos2B":
    #     #print(f"label: {label}")

    if label == "Opos1B_Oneg2B":
        #print(f"label: {label}")
        base_filter["polarity"] = "Negative"

    ##print("filter")
    #print(base_filter)

    # Perform similarity search
    relevant_docs = vector_database.similarity_search(
        query=question,
        filter=base_filter,
        k=10,  # Number of results to return
        fetch_k=96206,  # Number of results to fetch before filtering
    )

    #print("\nReview Id: ", review_id)
    #print("\n")

    #print("1:", relevant_docs)
    # Post-filtering to apply $ne condition on reviewId

    #NOTE: This filter applies to all labels.
    #Thats why we dont have any seperate if condition for Opos1B_Opos1B2 label.
    filtered_docs = [
        doc for doc in relevant_docs if doc.metadata["reviewId"] != review_id
    ]
    #print("2:", filtered_docs)
    if label == "Oneg1A_Opos2A" or label == "Opos1B_Opos2B" or label == "Opos1B_Oneg2B":
        filtered_docs = [
            doc for doc in relevant_docs if doc.metadata["aspect"] != aspect
        ]

    #print("3:", filtered_docs)
    if label == "Oneg1A_Opos1B":
        filtered_docs = [
            doc for doc in relevant_docs if doc.metadata["product_id"] != product_id
        ]
    #print("3:", filtered_docs)

    ##print("\nRelevant Doc in OURS:")
    #pprint.pprint(filtered_docs)

    # Generate the response using LLM
    answer = ""
    final_prompt = ""
    relevant_doc = ""
    if filtered_docs:
        relevant_doc = filtered_docs[0]
        final_prompt = prompt_in_chat_format.format(
            question=question, detailed_information=relevant_doc.page_content
        )

        # Generate answer using the large language model
        answer = llm(final_prompt)[0]["generated_text"]

    #print("########")

    return answer, final_prompt, relevant_doc


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
            "vanilla_rag_response",
            "vanilla_rag_prompt",
            "our_rag_response",
            "our_rag_prompt",
            "label",
            "our_relevant_doc",
        ]
        writer = csv.DictWriter(output_file_path, fieldnames=fieldnames)
        writer.writeheader()

        
        #for item in blocks_neg_100:
        for i in range(0, len(blocks_neg_100), 100):
            item = blocks_neg_100[i]
            print("\n\n")
            # Extract information
            question = item["question"]
            product_id = item["product_id"]
            label = item["label"]
            aspect = item["aspect"]
            answer = item["answer"]
            review_id = item["review_id"]
            # bought_together = item["bought_together"]
            print("Label: ",label)
            label = label_map[label]
            print("Label: ",label)
            # Save OpinionConv Response
            #print("# save OpinionConv Response")
            opinion_conv_response = answer

            # Save llm response
            print("save llm response")
            llm_response = get_llm_response(
                question, get_reader_model(), label, aspect, product_id
            )
            print("\n\n")
            # Save vanilla rag response
            #print("save vanilla rag response")
            # vanilla_rag_response, vanilla_rag_prompt = get_vanilla_rag_response(
            #     question, get_reader_model()
            # )
            #print("\n\n")
            # #Save our rag response
            #print("save our rag response")
            # our_rag_response, our_rag_prompt, relevant_doc = get_our_rag_response(
            #     question,
            #     label,
            #     aspect,
            #     product_id,
            #     review_id,
            #     answer,
            #     get_reader_model(),
            #     bought_together=[],
            # )

            # Write in csv
            writer.writerow(
                {
                    "query": question,
                    "opinion_conv_response": opinion_conv_response,
                    "llm_response": llm_response,
                    #"vanilla_rag_response": vanilla_rag_response,
                    #"vanilla_rag_prompt": vanilla_rag_prompt,
                    #"our_rag_response": our_rag_response,
                    #"our_rag_prompt": our_rag_prompt,
                    #"label": label,
                    #"our_relevant_doc": relevant_doc,
                }
            )

            # counter+=1
            # if counter>3:
            #     break
