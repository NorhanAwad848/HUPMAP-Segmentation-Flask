from routes.prediction_routes import prediction_route
from utilities.utility import prone_static_dir
from globals import HOST, PORT, DEBUG
from flask_cors import CORS
from flask import Flask


app = Flask(__name__, static_url_path='/static')
app.register_blueprint(prediction_route)
CORS(app)

if __name__ == '__main__':
    # pip install -r requirements.txt

    prone_static_dir("static/images")
    app.run(port=PORT, host=HOST, debug=DEBUG)
