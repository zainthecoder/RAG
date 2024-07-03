import json
import csv
import logging
import sys
import os
from langchain_community.vectorstores import FAISS
import pandas
from config import get_embedding_model, get_reader_model, conversation_mapping


vector_database = FAISS.load_local("faiss_index", get_embedding_model(), allow_dangerous_deserialization=True)

prompt_in_chat_format = """
Answer the question only based on the following context:
Give a short answer and don't mention "based on the provided context"
-----


Context: {context}

---

Answer the question based on the above context: 
Question: {question}
"""



def main_run():
    relevant_docs = vector_database.similarity_search(
                    query="Great battery life too if you know how to set your settings correctly.",
                    k=1
                )

    print(relevant_docs)


if __name__ == "__main__":
    main_run()

