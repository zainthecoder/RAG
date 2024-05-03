Development Branch: feat3

Add the script overview

the structure for the better understanding of the script

We are writing the script for 100_block_neg.json

	The Qa pairs we will create will be unique as the questions used are not used in 100_block_pos.json
	So the pair we need will always lie in the 	100_block_neg.

input_file_path = "/home/stud/abedinz1/localDisk/RAG/RAG/neg_qa_pairs.json"
	Regarding this we will be creating from 100_blocks_neg.json
	the script we use is query_bulk_generation.py


I need to create the vector database seperate from the everytime script run


While you screens are running, if you change code, it will affect all the screens.



##Command

python run_rag.py rag_output_1.json 1 -> means we are running the filtering



Flow:
1. Run the qa_pairs_bulk_generation.py
	-> qa_pairs
	-> unique ids
2. Run get_reviews_from_id.py
	-> filtered_reviews
	-> unique_product_ids
	-> add sentiment to the reviews
3. Run main.py
	-> create vector index from filtered reviews
4. Run run_rag.py
	-> run rag for every qa_pair

