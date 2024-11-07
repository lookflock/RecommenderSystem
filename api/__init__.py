
from flask import Flask
from .routes import recommender_api 

def create_app():
    # Initialize the Flask app
    app = Flask(__name__)
    
    # Register the blueprint
    app.register_blueprint(recommender_api, url_prefix='/api')
    
    return app
