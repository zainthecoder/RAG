import json
from vector_database import (
    vector_data_base_createion,
    retrieval_top_k,
    answer_with_rag,
    READER_LLM,
    RAG_PROMPT_TEMPLATE
)


# Define a function to load JSON data from a file
def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# Define a function to save JSON data to a file
def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

# Function to process questions and answers using RAG
def process_questions_and_answers(input_file_path, output_file_path):
    # Load JSON data from input file
    data = load_json(input_file_path)
    vector_database = vector_data_base_createion(docs_processed)
    # Iterate over each question and answer pair
    for item in data:
        question = item["question"]
        
        answer, relevant_docs = answer_with_rag(
            question, READER_LLM, vector_database
        )


        answer = answer
        item["answer_from_rag"] = answer

    # Save the updated data to a new JSON file
    save_json(data, output_file_path)
    print(f"Processed data saved to {output_file_path}")

# Define input and output file paths
input_file_path = "qa_pairs.json"
output_file_path = "rag_qa_pairs.json"

# Process questions and answers using RAG
process_questions_and_answers(input_file_path, output_file_path)
