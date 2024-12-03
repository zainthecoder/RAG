# Imports
from config import model_name, access_token
import csv
import pprint
from tqdm import tqdm
import re

from config import access_token, label_map, model_name, get_tokenizer, get_embedding_model, get_reader_model

# Load the CSV data
data = []
with open("output_file_path.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)


# Define a function to parse and extract only the assistant's response
def extract_assistant_response(full_response):
    # Use regex to find the text between the "assistant" header and the "<|eot_id|>"
    match = re.search(r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", full_response, re.DOTALL)
    if match:
        return match.group(1).strip()  # Return only the matched assistant response
    return full_response  # Return original if no match found

# Define a function to get the LLM response
def get_llm_response(messages, tokenizer, model):
    
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
        "cuda"
    )
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    generated_data = tokenizer.batch_decode(generated_ids)[0]

    #print("\nResonse")
    #pprint.pprint(generated_data)
    return generated_data


def prompt_creation(question, response1, response2, response3, response4):
    
    messages = [
    {
    "role": "user",
    "content": """
    You are an evaluator assigned to assess four response options provided to a customer's opinion/question.  
    Your task is to evaluate each response **holistically** based on key criteria and assign a single **overall score** between 1 and 5 for each option. 

    Please think critically, consider how each response meets the customer's needs, and justify the overall score with a brief explanation that touches on each criterion.

    ---

    ### **Evaluation Criteria**:

    1. **Realism**:  
       Does the response sound natural, engaging, and like a genuine human interaction?  
       - A high score (5) reflects a skilled salesperson who builds rapport and demonstrates empathy.  
       - A low score (1) indicates a response that feels robotic, detached, or artificial.

    2. **Relevance**:  
       How well does the response directly address the customer's opinion/question, needs, or concerns?  
       - A high score (5) means the response is tailored to the customer's situation.  
       - A low score (1) means it is off-topic or fails to address the customer’s intent.

    3. **Conciseness**:  
       Is the response clear and efficient in conveying necessary information without overwhelming the customer?  
       - A high score (5) indicates a well-balanced, concise response.  
       - A low score (1) indicates verbosity or excessive details.

    4. **Persuasiveness**:  
       How effectively does the response motivate the customer to take action or consider the product/service?  
       - A high score (5) reflects strong persuasive language and value propositions.  
       - A low score (1) lacks impact or fails to highlight benefits.

    5. **Subjectiveness**:  
       Does the response feel personalized or emotionally engaging?  
       - A high score (5) reflects individualized recommendations, opinions, or testimonials.  
       - A low score (1) feels generic, factual, or impersonal.

    ---

    ### **Evaluation Process**:

    1. **Individually assess** each response option based on the criteria listed.  
    2. Assign a single **overall score** between 1 and 5 to each option.  
    3. Provide a brief explanation for the score, referencing key strengths or weaknesses across the criteria.  
    4. Select the response with the highest overall score as the best option.

    ---

    ### **Evaluation Format**:

    Option 1:  
    - **Overall Score**: [score]  
    - **Justification**: [explain strengths and weaknesses based on the criteria]

    Option 2:  
    - **Overall Score**: [score]  
    - **Justification**: [explain strengths and weaknesses based on the criteria]

    Option 3:  
    - **Overall Score**: [score]  
    - **Justification**: [explain strengths and weaknesses based on the criteria]

    Option 4:  
    - **Overall Score**: [score]  
    - **Justification**: [explain strengths and weaknesses based on the criteria]

    The final answer is the option with the highest overall score: [chosen option].
                """,
            },
            {
                "role": "user",
                "content": f"""
    Customer opinion/question: {question}

    Options:  
    Option 1: {response1}  
    Option 2: {response2}  
    Option 3: {response3}  
    Option 4: {response4}

    Please provide a thoughtful, well-justified evaluation and select the best response based on the highest overall score.
                """,
            },
        ]
    return messages




# Process the data and write results to a new CSV
with open("rag_evaluation_file.csv", "w", newline="", encoding="utf-8") as output_file:
    fieldnames = [
        "question",
        "llm evaluation response",
        "opinion_conv_response",
        "llm_response",
        "vanilla_rag_response",
        "our_rag_response",
    ]

    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()  # Write the header row

    for item in tqdm(data, desc="Processing Queries"):
    #for i in range(0, len(data), 50):
        #item = data[i]
 
        question = item["query"]  # Assuming these keys match the CSV headers
        response1 = item["opinion_conv_response"]
        response2 = extract_assistant_response(item.get("our_rag_response", ""))
        response3 = extract_assistant_response(item["vanilla_rag_response"])
        response4 = extract_assistant_response(item["llm_response"])

        # Format the prompt
        final_prompt = prompt_creation(
            question=question,
            response1=response1,
            response2=response2,
            response3=response3,
            response4=response4,
        )

        # Get the LLM response
        llm_response = get_llm_response(
            final_prompt, get_tokenizer(), get_reader_model()
        )

        # Write the results to the CSV
        writer.writerow(
            {
                "question": question,
                "llm evaluation response": llm_response,
                "opinion_conv_response": response1,
               "our_rag_response": response2,
                "vanilla_rag_response": response3,
                "llm_response": response4,
            }
        )

print("Processing and CSV writing completed!")
