import json
import logging
import sys
import os
from langchain_community.vectorstores import FAISS
import pandas
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
def process_questions_and_answers(input_file_path, output_file_path, retrieval_filter):
    # Load JSON data from input file
    data = load_json(input_file_path)
    
    # Iterate over each question and answer pair
    for item in data:
        question = item["question"]
        product_id = item["key_question"].split('_')[0]
        question_key = item["key_question"]
        answer_key = item["key_answer"]

        answer, relevant_docs, final_prompt, rag_key = answer_with_rag(
            question, READER_LLM, vector_database, product_id, question_key, answer_key, retrieval_filter
        )

        if rag_key == answer_key:
            print("rag key is same as anaswer key")
            search_answer = "Not needed"
        else:    
            print("The rag answer is different, now we will search")
            search_pair = search_neg_qa_pairs(data, rag_key, question_key)
            search_answer = search_pair["answer"]

        # Update item with RAG results and search results
            item.update({
                "answer_from_rag": answer,
                "final_prompt": final_prompt,
                "search_answer": search_answer
            })
        logging.info(f"Relevant Document for the question: {question} are {relevant_docs}")
        save_json_append([item], output_file_path)  # Append mode

    logging.info(f"Processed data saved to {output_file_path}")


def answer_with_rag(
    question,
    llm,
    knowledge_index,
    product_id,
    question_key, 
    answer_key,
    retrieval_filter=False,
    num_retrieved_docs: int = 1,
    num_docs_final: int = 1,
):
    logging.info("=> Retrieving documents...")
    logging.info("question: %s", question)
    logging.info("productId: %s", product_id)

    # Set the default empty relevant page content
    relevant_page_content = ""

    # Retrieve documents based on the filter criteria
    try:
        if retrieval_filter:
            relevant_docs = knowledge_index.similarity_search(
                query=question,
                filter=dict(
                    productId=product_id,
                    sentiment="sentiment"
                ),
                k=num_retrieved_docs,
                fetch_k=1000
            )
        else:
            relevant_docs = knowledge_index.similarity_search(
                query=question,
                k=num_retrieved_docs
            )

        if relevant_docs:
            # Select the most relevant document
            relevant_doc = relevant_docs[0]
            relevant_page_content = relevant_doc.page_content
            metadata = relevant_doc.metadata
            rag_reviewer_id = metadata.get("reviewerID", None)
            rag_product_id = metadata.get("productId", None)
            rag_key = f"{rag_product_id}_{rag_reviewer_id}"
        else:
            logging.info(f"No relevant documents found")
            rag_key = ""

        final_prompt = prompt_in_chat_format.format(question=question, context=relevant_page_content)
        
        # Generate the answer using the language model
        answer = llm(final_prompt)[0]["generated_text"]

    except Exception as e:
        logging.info(f"Error during document retrieval or answer generation: {e}")
        answer, relevant_docs, final_prompt, rag_key = None, [], "", ""

    return answer, relevant_docs, final_prompt, rag_key



def main_run():
    """
    Processes questions and answers using the RAG model based on command line arguments.
    Expects at least one command line argument for filtering options.

    Usage:
        python script_name.py <filter_option>
    """
    logging.basicConfig(level=logging.INFO)

    # Check if filter option is provided
    if len(sys.argv) < 2:
        logging.error("No filter option provided. Usage: python script_name.py <filter_option>")
        sys.exit(1)

    retrieval_filter = sys.argv[1]
    logging.info("Filter: %s", retrieval_filter)

    base_dir = "/home/stud/abedinz1/localDisk/RAG/RAG"
    output_file_name = "rag_response_{}.txt".format(retrieval_filter)
    output_file_path = os.path.join(base_dir, "results", output_file_name)
    logging.info("Output file path: %s", output_file_path)

    input_file_path = os.path.join(base_dir, "data", "filtered_qa_pairs.json")

    # Process questions and answers
    try:
        process_questions_and_answers(input_file_path, output_file_path, retrieval_filter)
        logging.info("Processing completed successfully.")
    except Exception as e:
        logging.error("Failed to process questions and answers: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main_run()