import os
import json
from nanoid import generate
import pickle
import pprint
import ipdb

base_dir = "/home/s28zabed"

def extract_question_answer_pairs(data_stream):
    qa_pairs = []
    unique_ids = set()

    # for data in data_stream:
    #     ipdb.set_trace()
        # print("product_data")
        # pprint.pprint(product_data)
    for product_key, product_data in data_stream.items():
        print("product_key:", product_key)
        # Assuming product_data is a dictionary for each product
        for conv_type, conv_type_data in product_data.items():
            #print("conv_type", conv_type)
            # print("conv_type_data:",conv_type_data)
            if conv_type == "Qpos1A_Apos1A":
                question, answer = None, None
                for pair_key, pair_data in conv_type_data.items():
                    for key, value in pair_data.items():
                        if "Qpos1A" in key:
                            question = value["Question"]
                            product_id = value["Labels"]["Key"].split("_")[0]
                            aspect = value["Labels"]["Aspect"]
                            polarity = value["Labels"]["Polarity"]
                            review_id = value["Labels"]["Key"]
                            unique_ids.add(product_id)
                        elif "Apos1A" in key:
                            answer = value["Answer"]
                            product_id = value["Labels"]["Key"].split("_")[0]
                            aspect = value["Labels"]["Aspect"]
                            polarity = value["Labels"]["Polarity"]
                            review_id = value["Labels"]["Key"]
                            unique_ids.add(product_id)
                    if question and answer:
                        unique_id = generate(size=10)
                        qa_pairs.append(
                            {
                                "unique_id": unique_id,
                                "product_id": product_id,
                                "question": question,
                                "answer": answer,
                                "label": "Qpos1A_Apos1A",
                                "aspect": aspect,
                                "polarity": polarity,
                                "review_id": review_id,
                            }
                        )
            else:
                for pair_key, pair_data in conv_type_data.items():
                    question_flag = True
                    #print("pair_key:   ", pair_key)
                    #print("pair_data", pair_data)
                    for key, value in pair_data.items():
                        if question_flag:
                            question = value["Opinion"]
                            product_id = value["Labels"]["Key"].split("_")[0]
                            aspect = value["Labels"]["Aspect"]
                            polarity = value["Labels"]["Polarity"]
                            review_id = value["Labels"]["Key"]
                            if conv_type == "Oneg1A_Opos1B":
                                bought_together = value["Labels"]["bought_together"]
                            else:
                                bought_together = []
                            unique_ids.add(product_id)
                            question_flag = False
                        elif question_flag == False:
                            answer = value["Opinion"]
                            product_id = value["Labels"]["Key"].split("_")[0]
                            aspect = value["Labels"]["Aspect"]
                            polarity = value["Labels"]["Polarity"]
                            review_id = value["Labels"]["Key"]
                            unique_ids.add(product_id)

                    if question and answer:
                        unique_id = generate(size=10)
                        qa_pairs.append(
                            {
                                "unique_id": unique_id,
                                "product_id": product_id,
                                "question": question,
                                "answer": answer,
                                "label": conv_type,
                                "aspect": aspect,
                                "polarity": polarity,
                                "review_id": review_id,
                            }
                        )

    return qa_pairs, unique_ids


def load_jsonl_stream(file_path):
    """Load data from a large JSONL file line by line."""
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.")
        return []

    # Open the file and yield each line as a JSON object
    with open(file_path, "r") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise Error(e)  # Handle invalid JSON lines gracefully

def load_json(file_path):
    """Load data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.")
        return None

    with open(file_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None

# File path to your large JSONL file
#file_path = "/home/stud/abedinz1/localDisk/opinionconv-refactor/100_blocks_neg.jsonl"
file_path= base_dir+"/opinionconv-refactor/100_blocks_neg.json"

# Load data from the JSONL file as a stream
#data_stream = load_jsonl_stream(file_path)
data_stream = load_json(file_path)

# Extract question-answer pairs
qa_pairs, unique_ids = extract_question_answer_pairs(data_stream)

# Save qa_pairs as a Python list to a pickle file
output_file_path = (
    base_dir+"/RAG/data/neg_question_answer_pairs.pkl"
)
with open(output_file_path, "wb") as f:
    pickle.dump(qa_pairs, f)

# Save unique IDs to another pickle file
unique_ids_file_path = (
    base_dir+"/RAG/data/neg_unique_product_ids.pkl"
)
with open(unique_ids_file_path, "wb") as file:
    pickle.dump(unique_ids, file)
print("unique_ids", unique_ids)

print(f"Processed {len(qa_pairs)} neg question-answer pairs.")


# POS question answers
file_path= base_dir+"/opinionconv-refactor/100_blocks_pos.json"

# Load data from the JSONL file as a stream
#data_stream = load_jsonl_stream(file_path)
data_stream = load_json(file_path)

# Extract question-answer pairs
qa_pairs, unique_ids = extract_question_answer_pairs(data_stream)

# Save qa_pairs as a Python list to a pickle file
output_file_path = (
    base_dir+"/RAG/data/pos_question_answer_pairs.pkl"
)
with open(output_file_path, "wb") as f:
    pickle.dump(qa_pairs, f)

# Save unique IDs to another pickle file
unique_ids_file_path = (
    base_dir+"/RAG/data/pos_unique_product_ids.pkl"
)
with open(unique_ids_file_path, "wb") as file:
    pickle.dump(unique_ids, file)
print("unique_ids", unique_ids)

print(f"Processed {len(qa_pairs)} pos question-answer pairs.")