from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch

from flask_cors import CORS
# Load pre-trained model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings
app = Flask(__name__)
CORS(app)
@app.route('/api/get_embeddings', methods=['POST'])
def handle_get_embeddings():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    embeddings = get_embeddings(question)
    print(f"Embeddings {embeddings}")
    return jsonify({'embeddings': embeddings})

if __name__ == '__main__':
    app.run(debug=True)




