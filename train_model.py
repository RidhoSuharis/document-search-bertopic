# train_model.py
"""
Script ini digunakan untuk melatih model BERTopic pada data teks. 
Model dilatih menggunakan SentenceTransformer untuk embedding, UMAP untuk reduksi dimensi, 
dan HDBSCAN untuk mengelompokkan topik.

Setelah dilatih, model dan informasi terkait topik disimpan dalam bentuk file .pkl
agar bisa digunakan di aplikasi web tanpa melatih ulang model setiap kali dijalankan.
"""

from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
import pickle

# Step 1: Inisialisasi komponen yang akan digunakan dalam BERTopic
# ---------------------------------------------------------------
# Model Sentence Transformer akan menghasilkan embedding kalimat
sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# UMAP akan digunakan untuk reduksi dimensi dari embedding
umap_model = UMAP(
    n_neighbors=15,      # Jumlah tetangga terdekat untuk setiap titik (kecil untuk data topik yang tersegmentasi)
    n_components=5,      # Jumlah komponen output setelah reduksi dimensi
    min_dist=0.0,        # Kontrol seberapa rapat titik setelah reduksi dimensi
    metric='cosine'      # Ukuran kesamaan, sesuai dengan embedding dari Sentence Transformer
)

# HDBSCAN akan melakukan pengelompokan (clustering) topik dari data yang telah direduksi
hdbscan_model = HDBSCAN(
    min_cluster_size=15,          # Ukuran minimum sebuah cluster
    min_samples=5,
    metric='euclidean',           # Ukuran jarak untuk mengelompokkan data
    cluster_selection_method='eom',  # Metode pemilihan cluster yang lebih stabil
    prediction_data=True          # Aktifkan prediksi label untuk dokumen baru
)

# Vectorizer akan membuat representasi kata dari data teks yang sudah diolah
vectorizer_model = CountVectorizer(stop_words="english")

# Class-based TF-IDF Transformer
ctfidf_model = ClassTfidfTransformer()

# Step 2: Inisialisasi model BERTopic dengan komponen-komponen di atas
# ---------------------------------------------------------------
# BERTopic akan mengelompokkan dokumen menjadi topik-topik terpisah
model = BERTopic(
    language='english',
    top_n_words=10,               # Jumlah kata kunci untuk setiap topik
    min_topic_size=10,            # Ukuran minimum untuk sebuah topik (cluster)
    nr_topics=None,               # Jika None, BERTopic akan menentukan jumlah topik secara otomatis
    calculate_probabilities=True, # Aktifkan untuk menghitung probabilitas topik untuk setiap dokumen
    embedding_model=sentence_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    verbose=True                  # Untuk menampilkan log proses
)

# Step 3: Memuat dan menyiapkan data (contoh data)
# ---------------------------------------------------------------
data = [
    "An Automated Conversation System Using Natural Language Processing (NLP) Chatbot in Python",
    "Text Preprocessing for Text Mining in Organizational Research: Review and Recommendations",
    "The impact of preprocessing on word embedding quality: A comparative study",
    "A scoping review of preprocessing methods for unstructured text data to assess data quality",
    "Effectiveness of Preprocessing Algorithms for Natural Language Processing Applications",
    "Comparison of text preprocessing methods",
    "ChatGPT: Unlocking the Future of NLP in Finance",
    "Natural Language Processing (NLP) in Management Research: A Literature Review",
    "Resumate: A Prototype to Enhance Recruitment Process with NLP based Resume Parsing",
    "Enhancing Patient Experience by Automating and Transforming Free Text into Actionable Consumer Insights: A Natural Language Processing (NLP) Approach",
    "A deep learning-based model using hybrid feature extraction approach for consumer sentiment analysis",
    "NLP-based Feature Extraction for the Detection of COVID-19 Misinformation Videos on YouTube",
    "Feature Extraction and Analysis of Natural Language Processing for Deep Learning English Language",
    "A Comparative Analysis on Suicidal Ideation Detection Using NLP, Machine, and Deep Learning",
    "Ensemble Learning with Pre-Trained Transformers for Crash Severity Classification: A Deep NLP Approach",
    "Uncovering Semantic Inconsistencies and Deceptive Language in False News Using Deep Learning and NLP Techniques for Effective Management",
    "A novel hybrid approach of SVM combined with NLP and probabilistic neural network for email phishing",
    "Detection of Fake Job Postings by Utilizing Machine Learning and Natural Language Processing Approaches",
    "Applying NLP techniques to malware detection in a practical environment",
    "An ensemble machine learning approach through effective feature extraction to classify fake news"
]

# Step 4: Melatih Model
# ---------------------------------------------------------------
# Langkah ini mengubah data teks menjadi embedding, kemudian melatih model BERTopic
print("Encoding data...")
embeddings = sentence_model.encode(data, show_progress_bar=True)  # Menghasilkan embedding untuk setiap dokumen
print("Fitting BERTopic model...")
model.fit(data, embeddings)  # Melatih model dengan data dan embedding

# Step 5: Menyimpan Model dan Informasi Topik
# ---------------------------------------------------------------
# Setelah model dilatih, simpan model dan informasi topik ke dalam file .pkl
model_path = 'models/bertopic_model.pkl'
topic_info_path = 'models/topic_info.pkl'
document_topic_mapping_path = 'models/document_topic_mapping.pkl'  # Path untuk dokumen dan topik

# Simpan model BERTopic
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Simpan informasi topik
# Informasi ini akan mempermudah akses detail topik dalam aplikasi web
topic_info = model.get_topic_info()
with open(topic_info_path, 'wb') as f:
    pickle.dump(topic_info, f)

# Simpan mapping dokumen dan topik
document_topic_mapping = dict(zip(data, model.topics_))
with open(document_topic_mapping_path, 'wb') as f:
    pickle.dump(document_topic_mapping, f)

print(f"Model and topic info saved at '{model_path}', '{topic_info_path}', and '{document_topic_mapping_path}'.")