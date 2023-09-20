# app.py

from flask import Flask, render_template, request, jsonify
import json
import random

app = Flask(__name__)

metrics = {
    'loss': None,
    'accuracy': None,
    'batch_time': None
}

def update_precision_settings_file(precision_settings):
    with open('precision_settings.json', 'w') as file:
        json.dump(precision_settings, file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_metrics', methods=['POST'])
def update_metrics():
    data = request.get_json()
    metrics['loss'] = data['loss']
    metrics['accuracy'] = data['accuracy']
    metrics['batch_time'] = data['batch_time']
    return jsonify(success=True)

@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics)

precision_settings = {}

@app.route('/change_precision', methods=['GET'])
def change_precision():
    layer = request.args.get('layer')
    precision = request.args.get('precision')
    
    precision_settings[int(layer)] = precision
    update_precision_settings_file(precision_settings)

    return jsonify(precision_settings)

if __name__ == '__main__':
    app.run(debug=True)
