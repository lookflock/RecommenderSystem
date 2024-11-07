import firebase_admin
from firebase_admin import credentials, firestore

def initialize_firebase():
    # Check if Firebase has already been initialized to prevent re-initialization
    if not firebase_admin._apps:
        cred = credentials.Certificate('./firebase_credentials.json')
        firebase_admin.initialize_app(cred)
    
    # Return Firestore client
    return firestore.client()

# Initialize Firestore client
db = initialize_firebase()