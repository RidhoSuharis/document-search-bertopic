from flask import Flask, request, jsonify, render_template
import pickle
import os
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# 1. Load model dari file
model_path = 'models/bertopic_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Memuat mapping dokumen dan topik
with open('models/document_topic_mapping.pkl', 'rb') as f:
    document_topic_mapping = pickle.load(f)


# 2. Load SentenceTransformer untuk embedding
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('documents', [])
    if not data:
        return jsonify({'error': 'No documents provided'}), 400

    # Step 1: Encode documents
    embeddings = sentence_model.encode(data, show_progress_bar=True)

    # Step 2: Fit the BERTopic model
    topics, probs = model.transform(data, embeddings)

    # Return results
    return jsonify({
        'topics': topics,
        'probabilities': probs.tolist()
    })

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    results = []

    for doc, topic in document_topic_mapping.items():
        if query.lower() in doc.lower():  # Mencari dalam dokumen
            topic_info = model.get_topic(topic) if topic != -1 else "No Topic"
            results.append({
                "document": doc,
                "topic": " ".join(word[0] for word in topic_info) if topic_info else "No Topic",
                "topic_id": topic
            })

    return jsonify({"results": results})




@app.route('/topic_details', methods=['GET'])
def topic_details():
    topic_id = request.args.get('topic_id', type=int)
    if topic_id is None:
        return jsonify({'error': 'No topic_id provided'}), 400

    # Ambil detail vocab dari topic tertentu
    if topic_id not in model.get_topics():
        return jsonify({'error': 'Invalid topic_id'}), 404
    
    words = model.get_topic(topic_id)
    return jsonify({'topic_id': topic_id, 'words': words})

@app.route('/vocab', methods=['GET'])
def vocab():
    topic_id = request.args.get('topic_id', type=int)
    if topic_id is None:
        return jsonify({'error': 'No topic_id provided'}), 400

    # Mendapatkan kosakata untuk topik tertentu
    words = model.get_topic(topic_id) if topic_id in model.get_topics() else None
    if words is None:
        return jsonify({'error': 'Invalid topic_id'}), 404

    return jsonify({'topic_id': topic_id, 'words': words})


if __name__ == '__main__':
    app.run(debug=True)
