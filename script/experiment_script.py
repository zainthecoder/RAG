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

# #Comment this line when you dont have the vector database
vector_database = FAISS.load_local(
        "/home/stud/abedinz1/localDisk/RAG/RAG/script/faiss_index", get_embedding_model(), allow_dangerous_deserialization=True
    )
def transform_data(data):
    with open('/home/stud/abedinz1/localDisk/opinionconv-refactor/final_reviews_after_absa.json', 'r') as f:
        orignal_data = json.load(f)

    orignal_data_with_single_aspect = []

    for entry in orignal_data:
        aspects = entry.get("aspect", [])
        sentiments = entry.get("sentiment", [])
        
        # If there are multiple aspects or sentiments, create new entries for each
        max_len = max(len(aspects), len(sentiments))
        if max_len == 0:
            pass
        else:
            for i in range(max_len):
                new_entry = entry.copy()
                new_entry["aspect"] = aspects[i]
                new_entry["sentiment"] = sentiments[i]
                orignal_data_with_single_aspect.append(new_entry)
        
    return orignal_data_with_single_aspect

def get_llm_response(question, llm, label="", aspect="", product_id=""):
    detailed_information=""
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

    print("Creating Vector Database")

    with open("/home/stud/abedinz1/localDisk/opinionconv-refactor/final_reviews_after_absa.json", 'r') as file:
        data = json.load(file)
    
    # Transform the data
    transformed_data = transform_data(data)

    ds = Dataset.from_list(transformed_data)
    c=0
    for doc in ds:
        print("\n")
        pprint.pprint(doc)
        c+=1
        if c>5:
            break


    # """Preprocess documents for Langchain."""
    raw_knowledge_base = [
        LangchainDocument(
            page_content=doc["sentence"],
            metadata={
                "productId": doc["asin"],
                "aspect": doc["aspect"],
                "polarity": doc["sentiment"],
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

    print("question: ",question)
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


def get_our_rag_response(question, label, aspect, product_id, answer, llm):

    # create vector database
    if not os.path.exists("/home/stud/abedinz1/localDisk/RAG/RAG/script/faiss_index"):
        create_vector_database()


    if label == "Qpos1A_Apos1A":
        relevant_doc = vector_database.similarity_search(
        query=question, 
        k=1000,
        filter=dict(
            productId= product_id,
            aspect= aspect,
            polarity= 'Positive'
        )
        )
        
    elif label == "Oneg1A_Opos1A":

        relevant_doc = vector_database.similarity_search(
        query=question, 
        k=1000,
        filter=dict(
            productId= product_id,
            aspect= aspect,
            polarity= 'Positive'
        )
        )
        
    elif label == "Oneg1A_Opos1B":

        relevant_doc = vector_database.similarity_search(
        query=question, 
        k=1000,
        filter=dict(
            productId= {"$ne": product_id},
            aspect= aspect,
            polarity= 'Positive'
        )
        )
    
    print("Relevant Doc in OURS:")
    pprint.pprint(relevant_doc)
    answer = ""
    final_prompt=""
    if relevant_doc:
        relevant_doc = relevant_doc[0]
        relevant_page_content = relevant_doc.page_content
        final_prompt = prompt_in_chat_format.format(
            question=question, detailed_information=relevant_doc
        )

        # Generate the answer using the large language model
        answer = llm(final_prompt)[0]["generated_text"]
    return answer, final_prompt

if __name__ == '__main__':
    # Load pickled data
    with open("/home/stud/abedinz1/localDisk/RAG/RAG/data/question_answer_pairs.pkl", 'rb') as f:
        blocks_neg_100 = pickle.load(f)

    counter=0

    # Writing to CSV file
    with open("output_file_path.csv", "w", newline="", encoding="utf-8") as output_file_path:
        fieldnames = [
            "query",
            "opinion_conv_response",
            "llm_response",
            "vanilla_rag_response",
            "vanilla_rag_prompt",
            "our_rag_response",
            "our_rag_prompt"
        ]
        writer = csv.DictWriter(output_file_path, fieldnames=fieldnames)
        writer.writeheader()

        for item in blocks_neg_100:
            print("\n\n")

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
            llm_response = get_llm_response(question, get_reader_model(), label, aspect, product_id)

            #Save vanilla rag response
            print("save vanilla rag response")
            vanilla_rag_response, vanilla_rag_prompt = get_vanilla_rag_response(question, get_reader_model())

            # #Save our rag response
            print("save our rag response")
            our_rag_response, our_rag_prompt = get_our_rag_response(question, label, aspect, product_id, answer, get_reader_model())

            # Write in csv
            writer.writerow({
                "query": question,
                "opinion_conv_response": opinion_conv_response,
                "llm_response": llm_response,
                "vanilla_rag_response": vanilla_rag_response,
                "vanilla_rag_prompt": vanilla_rag_prompt,
                "our_rag_response": our_rag_response,
                "our_rag_prompt":our_rag_prompt
            })


            # counter+=1
            # if counter>5:
            #     break
