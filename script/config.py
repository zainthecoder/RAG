import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt
import locale
import json
import sys

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS

import pdb
from transformers import pipeline
import torch
import pprint
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# EMBEDDING_MODEL_NAME = "thenlper/gte-small"

# embedding_model = HuggingFaceEmbeddings(
#     model_name=EMBEDDING_MODEL_NAME,
#     multi_process=True,
#     model_kwargs={"device": "cuda"},
#     encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
# )

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# READER_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
# model = AutoModelForCausalLM.from_pretrained(
#     READER_MODEL_NAME, quantization_config=bnb_config, device_map="auto"
# )

# model.config.use_cache = False
# model.config.pretraining_tp = 1

# tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME,device_map="auto")

# READER_LLM = pipeline(
#         model=model,
#         tokenizer=tokenizer,
#         task="text-generation",
#         do_sample=True,
#         temperature=0.2,
#         repetition_penalty=1.1,
#         return_full_text=False,
#         max_new_tokens=500,
# )
# EMBEDDING_MODEL_NAME = "thenlper/gte-small"

# conversation_mapping = {
#     "Qpos1A_Apos1A": "Positive",
#     "Oneg1A_Opos1A": "Positive",
#     "Oneg1A_Opos1B": "Positive",
#     "Oneg1A_Opos2A": "Switch_Postive",
#     "Opos1B_Opos2B": "Switch_Positive",
#     "Opos1B_Oneg2B": "Switch_Negative"
# }
model_singleton = {}

def get_embedding_model():
    if 'embedding_model' not in model_singleton:
        EMBEDDING_MODEL_NAME = "thenlper/gte-small"
        model_singleton['embedding_model'] = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        )
    return model_singleton['embedding_model']

def get_reader_model():
    if 'reader_model' not in model_singleton:
        READER_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL_NAME, quantization_config=bnb_config, device_map="auto"
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME, device_map="auto")
        model_singleton['reader_model'] = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )
    return model_singleton['reader_model']

def get_tokenizer():
    if 'tokenizer' not in model_singleton:
        READER_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
        model_singleton['tokenizer'] = AutoTokenizer.from_pretrained(READER_MODEL_NAME, device_map="auto")
    return model_singleton['tokenizer']