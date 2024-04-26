import json
import os
import pprint
from nanoid import generate

#hi bro
def extract_question_answer_pairs(data):
    
    qa_pairs = []
    

    for product_key, product_data in data.items():
        #print(product_key)
        for conv_type, conv_type_data in product_data.items():
            #print(conv_type)
            #print("conv_type_data: ",conv_type_data)
            for conv_key, conv_data in conv_type_data.items():
                #pprint.pprint(conv_key)
                if conv_type == 'Qpos1A_Apos1A':
                    for pair_key, pair_data in conv_data.items():
                        #pprint.pprint(pair_key)
                        #pprint.pprint(pair_data)
                        if 'Qpos1A' in pair_key:
                            question = pair_data['Question']
                            key_question = pair_data['Labels']['Key']
                        elif 'Apos1A' in pair_key:
                            answer = pair_data['Answer']
                            key_answer = pair_data['Labels']['Key']

                    unique_id=generate(size=10) 
                    qa_pairs.append({
                            "unique_id":unique_id,
                            "key_question":key_question,
                            "key_answer":key_answer,
                            "question":question,
                            "answer":answer
                        })
                else:
                    counter = 1
                    for pair_key, pair_data in conv_data.items():
                        #pprint.pprint(pair_key)
                        #pprint.pprint(pair_data)
                        if counter == 1:
                            question = pair_data['Opinion']
                            key_question = pair_data['Labels']['Key']
                            counter += 1
                        elif counter == 2:
                            answer = pair_data['Opinion']
                            key_answer = pair_data['Labels']['Key']
                    unique_id=generate(size=10)
                    qa_pairs.append({
                            "unique_id":unique_id,
                            "key_question":key_question,
                            "key_answer":key_answer,
                            "question":question,
                            "answer":answer
                        })

    return qa_pairs

def load_json(file_path):
    """Load data from a JSON file."""
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found.")
        return []
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


   # File path to your JSON file
file_path = "/Users/zainabedin/Desktop/RAG/dataset.json"

    # Load data from JSON file
data = load_json(file_path)
print("zainnnn")
#pprint.pprint(data)
    # Create a Hugging Face dataset
qa_pairs = extract_question_answer_pairs(data)

# File path to save the Python list
file_path = "qa_pairs.json"

# Save qa_pairs as a Python list to a JSON file
with open(file_path, "w") as f:
    json.dump(qa_pairs, f)

 # "B00836Y6B2": {
 #        "Opos1B_Opos1B2_only_agreement": {
 #            "Opos1B_Opos1B2_1": {
 #                "Opos1B": {
 #                    "Opinion": "I was wondering if you have this phone B00836Y6B2, it might be a good choice because as far as I know about its price, Good product, great price.",
 #                        "Labels": {
 #                        "Key": "B00836Y6B2_AFGB723G8ECAX_0_0",
 #                            "Aspect": "price",
 #                                "Polarity": "positive"
 #                    }
 #                },
 #                "Opos1B2": {
 #                    "Opinion": "Yes, That's so true. This phone is also a good choice",
 #                        "Labels": {
 #                        "Key": "B00836Y6B2_AFGB723G8ECAX_0_0",
 #                            "Aspect": "price",
 #                                "Polarity": "positive"
 #                    }
 #                }
 #            },
 #   "B00836Y6B2": {
 #        "Qpos1A_Apos1A": {
 #            "Qpos1A_Apos1A_1": {
 #                "Qpos1A": {
 #                    "Question": "In your honest opinion, how is its price?",
 #                        "Labels": {
 #                        "Key": "B00836Y6B2_AFGB723G8ECAX_0_0",
 #                            "Aspect": "price",
 #                                "Polarity": "positive"
 #                    }
 #                },
 #                "Apos1A": {
 #                    "Answer": "Good product, great price.",
 #                        "Labels": {
 #                        "Key": "B00836Y6B2_AFGB723G8ECAX_0_0",
 #                            "Aspect": "price",
 #                                "Polarity": "positive"
 #                    }
 #                }
 #            },
