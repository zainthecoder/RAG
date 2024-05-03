import json

# Load data from the first JSON file
with open('pos_qa_pairs.json', 'r') as file:
    data1 = json.load(file)

# Load data from the second JSON file
with open('neg_qa_pairs.json', 'r') as file:
    data2 = json.load(file)

# Combine the data from both files into a single list
combined_data = data1 + data2

# Save the combined data to a new JSON file
with open('combined_qa_pairs.json', 'w') as file:
    json.dump(combined_data, file, indent=4)