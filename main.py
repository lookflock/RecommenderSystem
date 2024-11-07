from flask import Flask
from api.routes import recommender_api
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Register the recommender blueprint
app.register_blueprint(recommender_api, url_prefix='/api')

if __name__ == "__main__":
    app.run(debug=True)

