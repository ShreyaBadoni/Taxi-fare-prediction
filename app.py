from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load Random Forest model
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            distance = request.form.getlist("distance")
            duration = request.form.getlist("duration")
            speed = request.form.getlist("speed")

            df = pd.DataFrame({
                "TRIP_MILES": distance,
                "TRIP_SECONDS": duration,
                "TRIP_SPEED": speed
            }).astype(float)

            df["Predicted Fare"] = model.predict(df)
            result = df.to_html(index=False, classes="table")

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
