from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the model
model_file = "fine_tuned_co2_emission_model.pkl"
with open(model_file, "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('fuel.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        engine_size = float(data['engine_size'])
        cylinders = int(data['cylinders'])
        fuel_consumption = float(data['fuel_consumption'])

        # Prepare input for prediction
        features = np.array([[engine_size, cylinders, fuel_consumption]])
        prediction = model.predict(features)[0]

        return render_template('fuel.html', prediction=prediction)
    except Exception as e:
        return render_template('fuel.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
