from flask import Flask, jsonify, redirect, render_template, request
import os
from ai import aifunction

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def takeinput():
    if request.method == 'POST':
        userinput = request.form['userinput']
        return redirect(f"/generate?userinput={userinput}")  # Redirect to /generate with user input
    return render_template("index.html")

@app.route('/generate')
def generate_summary():
    userinput = request.args.get('userinput', '')
    results, predictions = aifunction(userinput)  # Call aifunction from ai.py

    # Pass results and predictions to the template for rendering
    return render_template("result.html", userinput=userinput, results=results, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
