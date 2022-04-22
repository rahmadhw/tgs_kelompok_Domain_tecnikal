from flask import Flask, render_template, request, jsonify, url_for
import joblib
import numpy as np



app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])

def predik():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        pred = [float(x) for x in request.form.values()]
        final_pred =[np.array(pred)]
        model,std_scaler = joblib.load("Modelling/model_joblib/model_data_fix.pkl")
        result = model.predict(final_pred)
        print(result)
        output = str(round(result[0], 2))
        return render_template("index.html", result = output)
    else:
        return "Error"


if __name__ == '__main__':
    app.run(port=5000, debug=True)