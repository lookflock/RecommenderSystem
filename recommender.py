import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from preprocess import preprocess_text
from model_utils import save_model, load_model

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
        
        save_model(self.tfidf_vectorizer, 'tfidf_vectorizer.pkl')
        save_model(self.product_vectors, 'product_vectors.pkl')
        save_model(self.products, 'products_data.pkl')

    def get_similar_products(self, product_ids, page_number=1, items_per_page=10):
        if isinstance(product_ids, str):
            product_ids = [product_ids]
        
        all_similarities = [
            cosine_similarity(self.product_vectors[self._get_index_by_id(pid)], self.product_vectors).flatten()
            for pid in product_ids if self._get_index_by_id(pid) is not None
        ]

        if not all_similarities:
            return []


        avg_similarities = np.mean(all_similarities, axis=0)
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

        return self._paginate_results(similar_products, page_number, items_per_page)

    def _get_index_by_id(self, product_id):
        return next((i for i, p in enumerate(self.products) if p['id'] == product_id), None)

    def _paginate_results(self, similar_products, page_number, items_per_page):
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
         
        return product_ids

    
    def load_model(self):
        self.tfidf_vectorizer = load_model('tfidf_vectorizer.pkl')
        self.product_vectors = load_model('product_vectors.pkl')
        self.products = load_model('products_data.pkl')
