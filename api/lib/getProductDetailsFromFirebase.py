from api.firebase_config import db
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
def getProductDetailsFromFirebase(product_ids):
    products_ref = db.collection('products')
    
    # Query products collection for all product IDs at once
    query = products_ref.where(filter=FieldFilter("id", "in", product_ids))

    docs = query.stream()
    
    # Collect and return product details
    product_details = [doc.to_dict() for doc in docs]
    
    return product_details
