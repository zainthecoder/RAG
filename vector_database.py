from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS

import pdb

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
)

def vector_data_base_createion(docs_processed):
    print(docs_processed)
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    return KNOWLEDGE_VECTOR_DATABASE

def retrieval_top_k(user_query, KNOWLEDGE_VECTOR_DATABASE):
    print(f"\nStarting retrieval for {user_query=}...")
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search_with_score(query=user_query, filter=dict(productId='098949232X'), k=1)

    if retrieved_docs:
        print(
            "\n==================================Top document=================================="
        )
        # Access the first element of the retrieved_docs list
        top_document = retrieved_docs[0][0]
        # Access the similarity score
        similarity_score = retrieved_docs[0][1]

        print(top_document.page_content)
        print("==================================Metadata==================================")
        # Access metadata of the top document
        print(top_document.metadata)
        print(f"Similarity score: {similarity_score}")
    else:
        print("No documents were retrieved for the given query and filter.")