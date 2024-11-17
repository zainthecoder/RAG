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
            You are an evaluator assigned to assess four response options provided to a customer question. 
            Please review each response in depth based on specific criteria and assign a score between 1 and 5 for each criterion.

            **Evaluation Criteria**:
            1. **Realism**:Assess how natural, personable, and engaging the sales agent's response feels. A high score (5) should indicate that the response sounds like a skilled salesperson genuinely interacting with the customer, building rapport, and showing empathy. A low score (1) indicates a response that feels artificial, overly rehearsed, or detached.
            2. **Relevance**: Check if the sales agent’s response directly addresses the customer’s needs, preferences, or concerns expressed in their question. A high score (5) means the response is precisely tailored to the customer's situation, demonstrating a deep understanding of their requirements. A low score (1) means the response is off-topic or does not align with the customer's expressed interests.
            3. **Conciseness**: Evaluate if the response provides the necessary information effectively without overwhelming the customer with excessive details. A high score (5) indicates the agent conveys key points clearly and efficiently, maintaining the customer’s attention. A low score (1) reflects a response that is too verbose or includes unnecessary or irrelevant information.
            4. **Persuasiveness**: Judge how effectively the response motivates the customer to consider or purchase the product/service. A high score (5) should indicate the use of persuasive language, compelling value propositions, and addressing objections skillfully. A low score (1) means the response lacks impact, fails to highlight benefits, or does not encourage action.
            5. **Subjectiveness**: Assess whether the response includes a personalized or emotionally engaging approach, such as sharing opinions, testimonials, or tailored recommendations. A high score (5) reflects a response that feels individualized and expressive, whereas a low score (1) indicates the response is entirely factual, generic, or devoid of personal touch.

            **Evaluation Process**:
            1. For each option, provide individual scores for the criteria listed above.
            2. Calculate the average score for each option based on the five criteria.
            3. Choose the option with the highest average score as the best response.

            Return the evaluation in the following format:
            Option 1:
              - Realism: [score]
              - Relevance: [score]
              - Conciseness: [score]
              - Persuasiveness: [score]
              - Subjectiveness: [score]
              - Average Score: [calculated average]
            Option 2: [scores and average]...
            ...
            The final answer is the option with the highest average score: [chosen option].
            """,
        },
        {
            "role": "user",
            "content": f"""
            Customer question: {question}

            Options:
            Option 1: {response1}
            Option 2: {response2}
            Option 3: {response3}
            Option 4: {response4}

            Please follow the specified format carefully, ensure each response is analyzed individually, and calculate the average before making the final selection.
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
