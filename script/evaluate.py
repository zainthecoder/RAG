# Imports
from config import model_name, access_token
import csv
import pprint
from tqdm import tqdm

from config import access_token, label_map, model_name, get_tokenizer, get_embedding_model, get_reader_model

# Load the CSV data
data = []
with open("output_file_path.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)


# Define a function to get the LLM response
def get_llm_response(messages, tokenizer, model):
    
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
        "cuda"
    )
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    generated_data = tokenizer.batch_decode(generated_ids)[0]

    print("\nResonse")
    pprint.pprint(generated_data)
    return generated_data


def prompt_creation(
    question,
    response1,
    response2,
    response3,
    response4,
):

    messages = [
        {
            "role": "user",
            "content": f"""
            You are a helpful and knowledgeable sales agent assisting a customer. Analyze and choose the best option for the following customer query:
            """,
        },
        {
            "role": "user",
            "content": f"""
            Customer question: {question}

            Options
            Option 1: {response1}
            Option 2: {response2}
            Option 3: {response3}
            Option 4: {response4}

            Ensure the final answer is clearly indicated by ending with {"The final answer is"}.
            """,
        },
    ]
    return messages


# Process the data and write results to a new CSV
with open("evaluation_file.csv", "w", newline="", encoding="utf-8") as output_file:
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

    #for item in tqdm(data, desc="Processing Queries"):
    for i in range(0, len(data), 50):
        item = data[i]
 
        question = item["query"]  # Assuming these keys match the CSV headers
        response1 = item["opinion_conv_response"]
        response2 = item["llm_response"]
        response3 = item["vanilla_rag_response"]
        response4 = item.get("our_rag_response", "")

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

        print("\n\nNew Question\n")
        print("final_prompt: ", final_prompt)
        print("llm_response:", llm_response)

        # Write the results to the CSV
        writer.writerow(
            {
                "question": question,
                "llm evaluation response": llm_response,
                "opinion_conv_response": response1,
                "llm_response": response2,
                "vanilla_rag_response": response3,
                "our_rag_response": response4,
            }
        )

print("Processing and CSV writing completed!")
