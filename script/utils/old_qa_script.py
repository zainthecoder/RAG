import json
import re

# Load the JSON data
with open('/Users/zainabedin/Desktop/RAG/dataset.json', 'r') as file:
    data = json.load(file)

# Create a dictionary to store the question-answer and opinion-answer pairs
all_pairs = {}

# Iterate through the data and extract the relevant information
for product_id, product_data in data.items():
    # Iterate through the Qpos1A_Apos1A section
    for qa_set_name, qa_set in product_data.items():
        if re.match(r'Q\w+_A\w+', qa_set_name):
            for qa_pair_name, qa_pair in qa_set.items():
                for qa_field_name, qa_field in qa_pair.items():
                    if re.match(r'Q\w+', qa_field_name):
                        question = qa_field['Question']
                    elif re.match(r'A\w+', qa_field_name):
                        answer = qa_field['Answer']
                all_pairs[question] = answer

    # Iterate through the Oneg1A_Opos1A section
    for opinion_set_name, opinion_set in product_data.items():
        if re.match(r'O\w+_O\w+', opinion_set_name):
            for opinion_pair_name, opinion_pair in opinion_set.items():
                for opinion_field_name, opinion_field in opinion_pair.items():
                    if re.match(r'O\w+', opinion_field_name):
                        question = opinion_field['Opinion']
                    elif re.match(r'O\w+', opinion_field_name):
                        answer = opinion_field['Opinion']
                all_pairs[question] = answer

# Print the combined question-answer and opinion-answer pairs
print("All Pairs:")
for question, answer in all_pairs.items():
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print()
