from flask import Flask
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)
auth = HTTPBasicAuth()

@auth.verify_password
def verify_password(username, password):
    if username == 'otherserver' and password == '123':
        return True
    else:
        return False

# Import the rest of the app
import upload_model
import deploy_model