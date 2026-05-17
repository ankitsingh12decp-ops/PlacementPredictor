"""
===================================================================
app.py — Flask Web Application for Placement Prediction
===================================================================
PURPOSE: A web app where users enter their details and get a
prediction of whether they'll be placed or not.

HOW FLASK WORKS (simply):
  1. Flask is a Python web framework (like a mini web server).
  2. We define "routes" — URLs that the user can visit.
  3. When a user visits a route, Flask runs a function and
     returns HTML to display in the browser.
  4. GET request = loading a page, POST request = submitting a form.

ROUTES:
  GET  /         → Show the input form
  POST /predict  → Process form data, make prediction, show result
===================================================================
"""

import os
from flask import Flask, render_template, request
from src.predict import load_model_and_scaler, make_prediction

# ---------------------------------------------------------------
# FLASK APP SETUP
# ---------------------------------------------------------------
# __name__ tells Flask where to find templates and static files
app = Flask(__name__)

# Load the ML model and scaler ONCE when the app starts
# (not on every request — that would be slow!)
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

try:
    model, scaler, feature_names = load_model_and_scaler(models_dir)
    print("✅ Model loaded successfully!")
except FileNotFoundError as e:
    print(f"⚠️ {e}")
    print("   Run 'python src/train_model.py' first to train the model.")
    model, scaler, feature_names = None, None, None


# ---------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------

@app.route("/")
def home():
    """
    Home page — displays the prediction form.

    render_template() loads the HTML file from templates/ folder
    and sends it to the browser.
    """
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle form submission — make a prediction.

    request.form contains all the data the user typed into the form.
    We extract each field, convert to float, and pass to our model.
    """
    try:
        if model is None:
            return render_template("index.html", prediction={
                "label": "⚠️ Model not loaded. Train the model first!",
                "probability": 0,
                "prediction": -1
            })

        # Extract form data and convert to numbers
        # request.form.get("field_name") gets the value from the HTML form
        features = {
            "cgpa": float(request.form.get("cgpa", 0)),
            "iq": float(request.form.get("iq", 100)),
            "communication_skill": float(request.form.get("communication_skill", 5)),
            "internships": float(request.form.get("internships", 0)),
            "projects": float(request.form.get("projects", 0)),
            "coding_skill": float(request.form.get("coding_skill", 5)),
            "attendance": float(request.form.get("attendance", 75)),
            "aptitude_score": float(request.form.get("aptitude_score", 50)),
            "logical_reasoning": float(request.form.get("logical_reasoning", 50)),
        }

        # Make prediction using our trained model
        result = make_prediction(model, scaler, features, feature_names)

        # Send the result back to the HTML template
        return render_template("index.html", prediction=result, inputs=features)

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return render_template("index.html", prediction={
            "label": f"⚠️ Error: {str(e)}",
            "probability": 0,
            "prediction": -1
        })


# ---------------------------------------------------------------
# RUN THE APP
# ---------------------------------------------------------------
if __name__ == "__main__":
    # debug=True: auto-reloads when you change code (dev only!)
    # host='0.0.0.0': accessible from other devices on the network
    print("\n🌐 Starting Flask server...")
    print("   Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
