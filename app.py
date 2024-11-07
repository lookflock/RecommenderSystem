import json
import time
import os
from recommender import ContentBasedRecommender

def load_products(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    start_time = time.time()
    products = load_products('arranged_products.json')
    
    recommender = ContentBasedRecommender()
    if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('product_vectors.pkl') and os.path.exists('products_data.pkl'):
        recommender.load_model()
    else:
        recommender.train(products)

    product_ids = ["3-pc-printed-lawn-suit-with-lawn-dupatta-ss-18-1-24-2-blue", "035"]
    page_number = 1
    items_per_page = 50

    similar_products = recommender.get_similar_products(product_ids, page_number, items_per_page)
    print(json.dumps(similar_products, indent=2))

    end_time = time.time()
    print(f"\nTotal program time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
