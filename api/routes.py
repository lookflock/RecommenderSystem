# api/routes.py
from flask import Blueprint, request, jsonify
from recommender import ContentBasedRecommender
from api.lib.getProductDetailsFromFirebase import getProductDetailsFromFirebase 

recommender_api = Blueprint('recommender_api', __name__)

recommender = ContentBasedRecommender()
recommender.load_model()  # Load the model once when the API starts

@recommender_api.route('/', methods=['GET'])
def root():
    return jsonify({"message": "Server is running"}), 200

@recommender_api.route('/train', methods=['POST'])
def train_model():
    products = request.json.get('products')
    recommender.train(products)
    return jsonify({'status': 'Training completed'}), 200

@recommender_api.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.json

    # Extract parameters from the request body
    product_ids = data.get('product_ids')
    page_number = data.get('page_number', 1)
    items_per_page = data.get('items_per_page', 10)

     # Fetch the recommendations using the recommender system
    recommendation_ids = recommender.get_similar_products(product_ids, page_number, items_per_page)

    # Get full product details from Firebase
    recommendations = getProductDetailsFromFirebase(recommendation_ids)

    return jsonify(recommendations), 200
