from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('car_price_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        input_features = [float(x) for x in request.form.values()]
        # Make prediction using the loaded model
        predicted_price = model.predict([input_features])
        return render_template('index.html', prediction_text=f'Predicted Car Price: ${predicted_price[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
