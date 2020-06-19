from flask import Flask, jsonify
import SentimentAnalyser

app = Flask(__name__)

@app.route("/")
def default():
    return "Hello world>>!!"

@app.route('/predictsentiment/<query>',methods=['GET'])
def predict(query):
    sentiment = SentimentAnalyser.returnSentiment(query)
    return jsonify(sentiment)

if __name__ == "__main__":
    app.run()
