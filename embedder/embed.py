from transformers import BertTokenizer, BertModel
import torch
import requests
import json

# Load pre-trained model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Function to generate embeddings for a given chunk of text
def get_embeddings(text, max_length=512):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    print("Chunks are chunks", chunks)
    embeddings = []
    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer(chunk_text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Assuming you want to take the mean of token embeddings for each chunk
        chunk_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        embeddings.append(chunk_embeddings)
    
    return embeddings

# Example: Read text data from a file
with open('mastery.txt', 'r') as file:
    text = file.read()

# Generate embeddings for the entire text
embeddings = get_embeddings(text)

# Supabase credentials
supabase_url = 'https://bydqlgirdhhxzgbwcjip.supabase.co'  # Replace with your Supabase URL
supabase_api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ5ZHFsZ2lyZGhoeHpnYndjamlwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjA1MTc4NjEsImV4cCI6MjAzNjA5Mzg2MX0.0kzcYdLpi3VdSTeQ7_YKkC4KkIlecnFoSBXqtegrTVc'  # Replace with your Supabase API key
table_name = 'documents'

# Prepare data to send to Supabase
data = {
    'content': text,
    'metadata': {'source': 'example'},
    'embedding': embeddings[0]  # Use only the first chunk's embeddings to match 768 dimensions
}

# Make a POST request to insert data into Supabase
try:
    response = requests.post(
        f"{supabase_url}/rest/v1/{table_name}",
        headers={
            'Content-Type': 'application/json',
            'apikey': supabase_api_key
        },
        json=data
    )

    # Check the response status
    if response.status_code == 201:
        print("Data inserted successfully.")
    else:
        print(f"Failed to insert data. Status code: {response.status_code}, Error message: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"Error connecting to Supabase: {e}")
