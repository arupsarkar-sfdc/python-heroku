from flask import Flask, request, redirect, jsonify
import joblib

app = Flask(__name__)
model=joblib.load('random_number_predictor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try: 
        data = request.get_json()
        X_new = [[data['X']]]
        prediction = model.predict(X_new)
        return jsonify({'prediction': str(prediction)})
    except:
        return jsonify({'trace': 'error'})
    
if __name__ == '__main__':
    app.run(debug=True)