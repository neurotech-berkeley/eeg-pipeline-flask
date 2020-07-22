from main import app, auth
from flask import request
from werkzeug.utils import secure_filename
from pathlib import Path

@app.route('/stream_connect', methods=['POST'])
@auth.login_required

def stream_connect():
    return