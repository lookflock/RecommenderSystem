import time
import re
import json
import os
import numpy as np
import faiss
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from joblib import Parallel, delayed
from scipy.sparse import save_npz, load_npz
from sklearn.decomposition import TruncatedSVD

# Ensure necessary NLTK data is downloaded
#nltk.download('stopwords')

# Preload stopwords globally
STOPWORDS = set(stopwords.words('english'))

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\W+', ' ', text.lower())
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

class ContentBasedRecommender:
    def __init__(self, max_similar_products=50, min_score=0.0, n_components=100):
        self.max_similar_products = max_similar_products
        self.min_score = min_score
        self.tfidf_vectorizer = TfidfVectorizer()
        self.product_vectors = None
        self.products = []
        self.n_components = n_components  # For dimensionality reduction with SVD
        self.index = None  # FAISS index for fast similarity search

    def train(self, products):
        # Store products and preprocess their descriptions with additional fields
        self.products = products

        # Parallel preprocessing of product descriptions
        processed_data = Parallel(n_jobs=-1)(delayed(preprocess_text)(
            f"{p.get('description', '')} {p.get('category', '')} {p.get('subCategory', '')} {p.get('subSubCategory', '')}"
        ) for p in products)

        # Create and fit TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_data)

        # Dimensionality reduction
        svd = TruncatedSVD(n_components=self.n_components)
        reduced_vectors = svd.fit_transform(tfidf_matrix)

        # Save vectorizer and product vectors
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open('svd_model.pkl', 'wb') as f:
            pickle.dump(svd, f)
        np.save('product_vectors.npy', reduced_vectors)
        with open('products_data.pkl', 'wb') as f:
            pickle.dump(self.products, f)

        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.n_components)  # Using Inner Product for cosine similarity
        self.index.add(reduced_vectors.astype('float32'))

    def get_similar_products(self, product_ids):
        if not isinstance(product_ids, list):
            product_ids = [product_ids]

        similar_products = []

        for product_id in product_ids:
        # Find product index
            idx = next((i for i, p in enumerate(self.products) if p['id'] == product_id), None)
            if idx is None:
                continue

        # Retrieve vector and search for similar products
            query_vector = np.load('product_vectors.npy')[idx].reshape(1, -1).astype('float32')
            distances, indices = self.index.search(query_vector, self.max_similar_products + 1)

        # Retrieve products based on indices and filter by min_score
        for i, score in zip(indices[0][1:], distances[0][1:]):  # Skip the product itself
            if score >= self.min_score:
                similar_products.append({
                    'id': self.products[i]['id'],
                    'name': self.products[i]['name'],
                    'score': float(score)  # Cast to Python float
                })

    # Sort by score in descending order
        similar_products.sort(key=lambda x: x['score'], reverse=True)
        return similar_products[:self.max_similar_products]


    def load_model(self):
        # Load vectorizer, SVD model, product vectors, and product data
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open('svd_model.pkl', 'rb') as f:
            svd = pickle.load(f)
        self.product_vectors = np.load('product_vectors.npy')
        with open('products_data.pkl', 'rb') as f:
            self.products = pickle.load(f)

        # Rebuild FAISS index
        self.index = faiss.IndexFlatIP(self.n_components)
        self.index.add(self.product_vectors.astype('float32'))

# Sample product data in the format you provided
start_time = time.time()

# Load products from JSON file
with open('products.json', 'r', encoding='utf-8') as f:
    products = json.load(f)

# Initialize and train recommender
recommender = ContentBasedRecommender()
if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('product_vectors.npy') and os.path.exists('products_data.pkl'):
    recommender.load_model()
else:
    recommender.train(products)

# Get recommendations for a product by ID
similar_products = recommender.get_similar_products(["14448588292256"])
print("Similar products:", json.dumps(similar_products, indent=2))
print("Number of similar products:", len(similar_products))
end_time = time.time()

# Calculate and print total program time
total_time = end_time - start_time
print(f"\nTotal program time: {total_time:.2f} seconds")
