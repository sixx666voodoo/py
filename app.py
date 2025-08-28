from flask import Flask, render_template, request
import secrets

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def order():
    if request.method == 'POST':
        email = request.form.get('email')
        code = secrets.token_hex(4).upper()
        return render_template('confirm.html', email=email, code=code)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
