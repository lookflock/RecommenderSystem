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
from collections import defaultdict

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\W+', ' ', text.lower())
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

class ContentBasedRecommender:
    def __init__(self, max_similar_products=50000, min_score=0.01):
        self.max_similar_products = max_similar_products
        self.min_score = min_score
        self.tfidf_vectorizer = TfidfVectorizer()
        self.product_vectors = None
        self.products = []
        
    def train(self, products):
        self.products = products
        processed_data = [
            preprocess_text(
                f"{p.get('description', '')} {p.get('category', '')} {p.get('subCategory', '')} {p.get('subSubCategory', '')}"
            )
            for p in products
        ]
        self.product_vectors = self.tfidf_vectorizer.fit_transform(processed_data)
        
        with open('tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open('product_vectors.pkl', 'wb') as f:
            pickle.dump(self.product_vectors, f)
        with open('products_data.pkl', 'wb') as f:
            pickle.dump(self.products, f)
    
    def get_similar_products(self, product_ids, page_number=1, items_per_page=10):
        if isinstance(product_ids, str):
            product_ids = [product_ids]
        
        all_similarities = []
        
        for product_id in product_ids:
            idx = next((i for i, p in enumerate(self.products) if p['id'] == product_id), None)
            if idx is None:
                continue
            
            cosine_similarities = cosine_similarity(self.product_vectors[idx], self.product_vectors).flatten()
            all_similarities.append(cosine_similarities)
        
        if not all_similarities:
            return {
                'products': [],
                'total_count': 0,
                'page_number': page_number,
                'items_per_page': items_per_page,
                'total_pages': 0
            }

        avg_similarities = sum(all_similarities) / len(all_similarities)
        similar_indices = avg_similarities.argsort()[-self.max_similar_products - 1:-1][::-1]
        
        similar_products = [
            {
                'id': self.products[i]['id'],
                'name': self.products[i]['name'],
                'supplier': self.products[i]['supplier'],
                'score': avg_similarities[i]
            }
            for i in similar_indices if avg_similarities[i] >= self.min_score
        ]
        
        supplier_groups = defaultdict(list)
        for product in similar_products:
            supplier_groups[product['supplier']].append(product)
        
        diverse_products = []
        while len(diverse_products) < len(similar_products):
            for supplier, products in supplier_groups.items():
                if products:
                    diverse_products.append(products.pop(0))
                    if len(diverse_products) >= len(similar_products):
                        break

        total_count = len(diverse_products)
        start_index = (page_number - 1) * items_per_page
        end_index = min(start_index + items_per_page, total_count)
        
        paginated_products = diverse_products[start_index:end_index]
        product_ids = [product['id'] for product in paginated_products]
         
        # return {
        #     #'products': paginated_products,
        #     'product_ids': product_ids,
        #     #'total_count': total_count,
        #     #'page_number': page_number,
        #     #'items_per_page': items_per_page,
        #     #'total_pages': -(-total_count // items_per_page)
        # }
        return product_ids
    
    def load_model(self):
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open('product_vectors.pkl', 'rb') as f:
            self.product_vectors = pickle.load(f)
        with open('products_data.pkl', 'rb') as f:
            self.products = pickle.load(f)

start_time = time.time()

with open('arranged_products.json', 'r', encoding='utf-8') as f:
    products = json.load(f)

recommender = ContentBasedRecommender()
if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('product_vectors.pkl') and os.path.exists('products_data.pkl'):
    recommender.load_model()
else:
    recommender.train(products)

product_ids = ["3-pc-printed-lawn-suit-with-lawn-dupatta-ss-18-1-24-2-blue","035"]
page_number =2
items_per_page = 50

similar_products = recommender.get_similar_products(product_ids, page_number, items_per_page)
print(json.dumps(similar_products, indent=2))

end_time = time.time()

total_time = end_time - start_time
print(f"\nTotal program time: {total_time:.2f} seconds")
