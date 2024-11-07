from flask import Flask
from api.routes import recommender_api
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials
from firebase_functions import https_fn

app = Flask(__name__)
CORS(app)

# Register the recommender blueprint
app.register_blueprint(recommender_api, url_prefix='/api')

# Initialize Firebase Admin SDK
cred = credentials.Certificate('./firebase_credentials.json')  
firebase_admin.initialize_app(cred)

# Define the entry point for your Cloud Function
@https_fn.on_request()
def main(req: https_fn.Request) -> https_fn.Response:
    # Handle the request using your Flask app
    with app.test_request_context(req.raw_body.decode('utf-8'), method=req.method, headers=req.headers):
        response = app.full_dispatch_request()
        return https_fn.Response(response.data, response.status_code, response.headers)
