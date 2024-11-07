import time
import re
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pickle
# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')

def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Lowercase and remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text.lower())
    
    # Tokenize and remove stopwords
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

class ContentBasedRecommender:
    def __init__(self, max_similar_products=50, min_score=0.2):
        self.max_similar_products = max_similar_products
        self.min_score = min_score
        self.tfidf_vectorizer = TfidfVectorizer()
        self.product_vectors = None
        self.products = []
        
    def train(self, products):
        # Store products and preprocess their descriptions
        self.products = products
        processed_descriptions = [preprocess_text(p['description']) for p in products]
        
        # Create TF-IDF matrix for product descriptions
        self.product_vectors = self.tfidf_vectorizer.fit_transform(processed_descriptions)
        
        # Save vectorizer and product vectors for future use
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open('product_vectors.pkl', 'wb') as f:
            pickle.dump(self.product_vectors, f)
        with open('products_data.pkl', 'wb') as f:
            pickle.dump(self.products, f)
    
    def get_similar_products(self, product_id):
        # Find the index of the product by ID
        idx = next((i for i, p in enumerate(self.products) if p['id'] == product_id), None)
        if idx is None:
            return []
        
        # Compute cosine similarity between this product and all others
        cosine_similarities = cosine_similarity(self.product_vectors[idx], self.product_vectors).flatten()
        
        # Find and rank similar products
        similar_indices = cosine_similarities.argsort()[-self.max_similar_products - 1:-1][::-1]
        similar_products = [
            {
                'id': self.products[i]['id'],
                'name': self.products[i]['name'],
                'score': cosine_similarities[i]
            }
            for i in similar_indices if cosine_similarities[i] >= self.min_score
        ]
        
        return similar_products


    def load_model(self):
        # Load vectorizer, product vectors, and product data
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open('product_vectors.pkl', 'rb') as f:
            self.product_vectors = pickle.load(f)
        with open('products_data.pkl', 'rb') as f:
            self.products = pickle.load(f)
# Sample product data in the format you provided
start_time = time.time()
# Load products from JSON file
with open('products.json', 'r', encoding='utf-8') as f:
    products = json.load(f)


# Initialize and train recommender
recommender = ContentBasedRecommender()
if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('product_vectors.pkl') and os.path.exists('products_data.pkl'):
    recommender.load_model()
else:
    recommender.train(products)
    
# Get recommendations for a product by ID
similar_products = recommender.get_similar_products("14448588292256")
print("Similar products:", json.dumps(similar_products, indent=2))

end_time = time.time()

# Calculate and print total program time
total_time = end_time - start_time
print(f"\nTotal program time: {total_time:.2f} seconds")