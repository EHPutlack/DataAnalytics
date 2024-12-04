from flask import Flask, render_template, request
app = Flask('app')

@app.route('/')
def hello_world():
  return 'Hello, World!'

app.run(host= '0.0.0.0', port=8080)