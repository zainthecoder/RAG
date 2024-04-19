import json

def extract_question_answer_pairs(data):
    questions = []
    answers = []

    for product_key, product_data in data.items():
        for conv_type, conv_type_data in product_data.items():
            if conv_type.startswith('conv_type_'):
                for conv_key, conv_data in conv_type_data.items():
                    if isinstance(conv_data, dict):
                        for pair_key, pair_data in conv_data.items():
                            if isinstance(pair_data, dict):
                                # Check if the pair contains 'Question' and 'Answer' keys
                                if 'Qpos1A' in pair_data and 'Apos1A' in pair_data:
                                    questions.append(pair_data['Qpos1A']['Question'])
                                    answers.append(pair_data['Apos1A']['Answer'])
                                elif 'Oneg1A' in pair_data and 'Opos1A' in pair_data:
                                    questions.append(pair_data['Oneg1A']['Opinion'])
                                    answers.append(pair_data['Opos1A']['Opinion'])

    return questions, answers

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
