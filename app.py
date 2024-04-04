from flask import Flask, redirect, render_template, request
from ai import aifunction  # Import the aifunction from ai.py

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
    prediction = aifunction(userinput)  # Call aifunction from ai.py to get the prediction

    # Pass prediction to the template for rendering
    return render_template("result.html", userinput=userinput, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
