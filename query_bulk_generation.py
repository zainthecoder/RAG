import json

def extract_question_answer_pairs(data):
    questions = []
    answers = []
    qa_pairs=[]

    for product_key(B00836Y6B2), product_data in data.items():
        for conv_type(Opos1B_Opos1B2_only_agreement), conv_type_data in product_data.items():
            
            for conv_key(Opos1B_Opos1B2_1), conv_data in conv_type_data.items():
                    if conv_type == 'Qpos1A_Apos1A':
                        for pair_key("Opos1B"), pair_data in conv_data.items():
                            

                                if 'Qpos1A' in pair_key:
                                    question = pair_data[pair_key][Question]
                                    key_question =  pair_data[pair_key][Labels][Key]
                                elif 'Apos1A' in pair_key:
                                    answer = pair_data[pair_key][Answer]
                                    key_answer =  pair_data[pair_key][Labels][Key]

                               # qa_pairs.append((key_question, key_answer, question, answer))
                               qa_pairs.append((
                                key_question,
                                key_answer,
                                question,
                                answer
                            ))

                    else:
                        counter = 1
                        for pair_key("Opos1B"), pair_data in conv_data.items():
                            

                                if counter==1:
                                    question = pair_data[pair_key][Opinion]
                                    key_question =  pair_data[pair_key][Labels][Key]
                                    counter++;
                                elif counter ==2:
                                    answer = pair_data[pair_key][Opinion]
                                    key_answer =  pair_data[pair_key][Labels][Key]

                                qa_pairs.append((
                                key_question,
                                key_answer,
                                question,
                                answer
                            ))


    return qa_pairs

 "B00836Y6B2": {
        "Opos1B_Opos1B2_only_agreement": {
            "Opos1B_Opos1B2_1": {
                "Opos1B": {
                    "Opinion": "I was wondering if you have this phone B00836Y6B2, it might be a good choice because as far as I know about its price, Good product, great price.",
                        "Labels": {
                        "Key": "B00836Y6B2_AFGB723G8ECAX_0_0",
                            "Aspect": "price",
                                "Polarity": "positive"
                    }
                },
                "Opos1B2": {
                    "Opinion": "Yes, That's so true. This phone is also a good choice",
                        "Labels": {
                        "Key": "B00836Y6B2_AFGB723G8ECAX_0_0",
                            "Aspect": "price",
                                "Polarity": "positive"
                    }
                }
            },
   "B00836Y6B2": {
        "Qpos1A_Apos1A": {
            "Qpos1A_Apos1A_1": {
                "Qpos1A": {
                    "Question": "In your honest opinion, how is its price?",
                        "Labels": {
                        "Key": "B00836Y6B2_AFGB723G8ECAX_0_0",
                            "Aspect": "price",
                                "Polarity": "positive"
                    }
                },
                "Apos1A": {
                    "Answer": "Good product, great price.",
                        "Labels": {
                        "Key": "B00836Y6B2_AFGB723G8ECAX_0_0",
                            "Aspect": "price",
                                "Polarity": "positive"
                    }
                }
            },


# Load the JSON file
with open("/content/drive/MyDrive/RAG/dataset.json", "r") as f:
    json_data = json.load(f)

# Extract question-answer pairs
questions, answers = extract_question_answer_pairs(json_data)

# Print the question-answer pairs
for q, a in zip(questions, answers):
    print("Question:", q)
    print("Answer:", a)
    print()

def extract_qa_pairs(data, outer_key=None):
    qa_pairs = []

    for key, value in data.items():
        if key == "Qpos1A_Apos1A":
            for pair in value.values():
                question = pair["Qpos1A"]["Question"]
                answer = pair["Apos1A"]["Answer"]
                qa_pairs.append((outer_key, question, answer))

        elif key.startswith("Oneg1A_Opos"):
            for pair in value.values():
                question = pair["Oneg1A"]["Opinion"]
                answer = pair["Opos1A"]["Opinion"]
                qa_pairs.append((outer_key, question, answer))

        elif isinstance(value, dict):
            qa_pairs.extend(extract_qa_pairs(value, key))

    return qa_pairs

# Example usage
data = {
    "B00J9XQRFG": {
        "Qpos1A_Apos1A": {},
        "Oneg1A_Opos1A": {},
        "Oneg1A_Opos1B_retrieved": {},
        # ... other keys
    },
    "B00836Y6B2": {
        "Qpos1A_Apos1A": {
            # ... question-answer pairs
        },
        "Oneg1A_Opos1A": {
            # ... opinion-opinion pairs
        },
        # ... other keys
    },
    # ... other outer keys
}

qa_pairs = extract_qa_pairs(data)

for outer_key, question, answer in qa_pairs:
    print(f"Outer Key: {outer_key}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print()