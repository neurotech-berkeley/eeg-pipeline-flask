from main import app, auth
from flask import request
from werkzeug.utils import secure_filename
from pathlib import Path

@app.route('/upload_model', methods=['POST'])
@auth.login_required
def upload_model():
    f = request.files['model']
    Path('./saved_models/' + secure_filename(request.form['user'])).mkdir(parents=True, exist_ok=True)
    f.save('./saved_models/' + secure_filename(request.form['user']) + '/' + secure_filename(f.filename))
    return "Model uploaded"