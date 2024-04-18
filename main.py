from tqdm.notebook import tqdm
from tqdm.gui import tqdm
import pandas as pd
from typing import List, Tuple
from datasets import Dataset
import locale
import os
import json
import pdb
from vector_database import vector_data_base_createion, retrieval_top_k, answer_with_rag, READER_LLM, RAG_PROMPT_TEMPLATE
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter



def load_json(file_path: str) -> List[dict]:
    """Load data from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        return data
    else:
        print(f"File '{file_path}' not found.")
        return []


def create_huggingface_dataset(data: List[dict]) -> Dataset:
    """Create a Hugging Face dataset from JSON data."""
    review_texts = [item["reviewText"] for item in data]
    product_ids = [item["asin"] for item in data]
    ds = Dataset.from_dict({"reviewText": review_texts, "asin": product_ids})
    return ds


def preprocess_documents(ds: Dataset) -> List[LangchainDocument]:
    """Preprocess documents for Langchain."""
    raw_knowledge_base = [
        LangchainDocument(page_content=doc["reviewText"], metadata={"productId": doc["asin"]})
        for doc in tqdm(ds)
    ]
    return raw_knowledge_base


def split_documents(raw_docs: List[LangchainDocument]) -> List[LangchainDocument]:
    """Split documents using Langchain's RecursiveCharacterTextSplitter."""
    markdown_separators = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True,
        strip_whitespace=True,
        separators=markdown_separators,
    )

    docs_processed = []
  
    for doc in raw_docs:
        docs_processed += text_splitter.split_documents([doc])
    return docs_processed


def main():
    pd.set_option("display.max_colwidth", None)
    locale.getpreferredencoding = lambda: "UTF-8"
    #pdb.set_trace()

    # File path to your JSON file
    file_path = "/home/s28zabed/RAG/reviews.json"

    # Load data from JSON file
    data = load_json(file_path)

    # Create a Hugging Face dataset
    ds = create_huggingface_dataset(data)
    print(ds)

    # Preprocess documents for Langchain
    raw_docs = preprocess_documents(ds)

    # Split documents using Langchain's text splitter
    docs_processed = split_documents(raw_docs)

    # Print the first processed document
    print(docs_processed)
    user_query = "I have heard that the quality of the product is not that good"

    vector_database = vector_data_base_createion(docs_processed)
    retrieval_top_k(user_query, vector_database)
    print ("######zain####")

    question = "is this a good calendar?"

    answer, relevant_docs = answer_with_rag(
        question, READER_LLM, vector_database
    )
    print("==================================Answer==================================")
    print(f"{answer}")
    print("==================================Source docs==================================")
    for i, doc in enumerate(relevant_docs):
        print(f"Document {i}------------------------------------------------------------")
        print(doc)



if __name__ == "__main__":
    main()


