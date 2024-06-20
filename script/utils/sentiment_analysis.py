from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn.functional as F

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Maximum sequence length allowed by the BERT model
MAX_SEQ_LENGTH = 512

def preprocess_text(text):
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    
    # Truncate or pad the sequence to the maximum length
    tokens = tokens[:MAX_SEQ_LENGTH - 2]  # -2 for [CLS] and [SEP] tokens
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Pad input IDs if necessary
    input_ids = pad_sequence([torch.tensor(input_ids)], batch_first=True, padding_value=tokenizer.pad_token_id)
    
    return input_ids

def predict_sentiment(text):
    # Preprocess the text
    input_ids = preprocess_text(text)
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient calculation
    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=1)
    
    # Get predicted label (0 for negative, 1 for positive)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    
    # Convert label to sentiment
    sentiment = "positive" if predicted_label == 1 else "negative"
    
    return sentiment

