from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello Docker!, Hello, world!-updated new 1"

@app.route('/about')
def about():
    return "This is the about page."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = True)