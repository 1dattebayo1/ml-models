from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the form
        input_data = [float(request.form.get(f"feature_{i}")) for i in range(60)]
        input_data_as_numpy_array = np.array(input_data).reshape(1, -1)

        # Make a prediction using the loaded model
        prediction = model.predict(input_data_as_numpy_array)

        # Determine the result
        result = "Rock" if prediction[0] == "R" else "Mine"

        return render_template("index.html", result=result)
    except Exception as e:
        return render_template("index.html", error="An error occurred. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
