import json

# Function to load JSON data from a file and find the first 200 unique productIds
def find_first_200_unique_product_ids(path):
    unique_product_ids = set()
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for obj in data:
            product_id = obj.get("productId")
            if product_id:
                unique_product_ids.add(product_id)
                if len(unique_product_ids) >= 200:
                    break
    return list(unique_product_ids)

# Example usage
path_to_file = "/Users/zainabedin/Desktop/RAG/100_blocks_pos.json"
first_200_product_ids = find_first_200_unique_product_ids(path_to_file)
print(first_200_product_ids)
