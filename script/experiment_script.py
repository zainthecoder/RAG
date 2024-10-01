import json
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores.utils import DistanceStrategy
from datasets import Dataset
import os

from config import get_embedding_model, get_reader_model, conversation_mapping



#vector store for vanilla rag
vainlla_vector_database = FAISS.load_local("faiss_index", get_embedding_model(), allow_dangerous_deserialization=True)

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

def get_llm_response(question, label="", aspect="", product_id=""):

    if label == "Qpos1A_Apos1A":
        detailed_information = f"The answer should have positive polarity about aspect: {aspect} and about same product with product id: {product_id}"
    elif label == "Oneg1A_Opos1A":
        detailed_information = f"The answer should have  positive polarity about aspect: {aspect} and about same product with product id: {product_id}"
    elif label == "Oneg1A_Opos1B":
        detailed_information = f"The answer should have  positive polarity about aspect: {aspect} but with different from product with product id: {product_id}"
    
    final_prompt = prompt_in_chat_format.format(question=question, detailed_information=detailed_information)

    # Generate the answer using the large language model
    answer = get_reader_model(final_prompt)[0]["generated_text"]
    return answer

def create_vector_database(reviews_input_file_path):
    #Load data using hugginface dataset


    #"""Preprocess documents for Langchain."""
    raw_knowledge_base = [
        LangchainDocument(
            page_content=doc["reviewText"], 
        )
        for doc in ds
    ]
    
    db = FAISS.from_documents(
        raw_knowledge_base, get_embedding_model(), distance_strategy=DistanceStrategy.COSINE
    )
    print("saving the data index")
    db.save_local("faiss_index")
    
def create_vector_database(reviews_input_file_path):
    #Load data using hugginface dataset


    #"""Preprocess documents for Langchain."""
    raw_knowledge_base = [
        LangchainDocument(
            page_content=doc["reviewText"], 
        )
        for doc in ds
    ]
    
    db = FAISS.from_documents(
        raw_knowledge_base, get_embedding_model(), distance_strategy=DistanceStrategy.COSINE
    )
    print("saving the data index")
    db.save_local("faiss_index")
    

def get_vanilla_rag_response(question):

    #create vector database
    if not os.path.exists("index path"):
        create_vector_database("")

    vector_database = FAISS.load_local("faiss_index", get_embedding_model(), allow_dangerous_deserialization=True)


    relevant_doc = vector_database.similarity_search(
        query=question,
        k=1
    )    
    relevant_doc = relevant_docs[0]
    relevant_page_content = relevant_doc.page_content
    final_prompt = prompt_in_chat_format.format(question=question, detailed_information=relevant_doc)

    # Generate the answer using the large language model
    answer = get_reader_model(final_prompt)[0]["generated_text"]
    return answer

def get_our_rag_response(question, label, aspect, product_id):

    #create vector database
    if not os.path.exists("index path"):
        create_our_vector_database("", )

    vector_database = FAISS.load_local("faiss_index", get_embedding_model(), allow_dangerous_deserialization=True)


    relevant_doc = vector_database.similarity_search(
        query=question,
        k=1
    )    
    relevant_doc = relevant_docs[0]
    relevant_page_content = relevant_doc.page_content
    final_prompt = prompt_in_chat_format.format(question=question, detailed_information=relevant_doc)

    # Generate the answer using the large language model
    answer = get_reader_model(final_prompt)[0]["generated_text"]
    return answer


data = load_json(question_answer_pairs)

for item in data:
    
    # extract information from object 
    question = item["question"]
    product_id = item["key_question"].split('_')[0]  #BUG
    question_key = item["key_question"]
    answer_key = item["key_answer"]
    label = item["label"]
    aspect = item["aspect"]
    answer = item["answer"]

    #save OpinionConv Response
    opinion_conv_response = answer

    #save llm response
    llm_response = get_llm_response(question, label, aspect, product_id)
    
    #save vanilla rag response
    vanilla_rag_response = get_vanilla_rag_response(question)

    #save our rag response
    our_rag_response = get_our_rag_response(question, label, aspect, product_id)