from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load_model('sentiment_analysis_model.h5')

# Load the saved tokenizer
loaded_tokenizer = Tokenizer()
loaded_tokenizer.word_index = joblib.load('sentiment_analysis_tokenizer.pkl')
loaded_tokenizer.index_word = dict((i, w) for w, i in loaded_tokenizer.word_index.items())

# Define the maximum number of words to use in the model
max_words = 10000

# Define the maximum length of a review
max_length = 100

# Function to make predictions on new data
def prediction(new_data):
    new_data = [loaded_tokenizer.texts_to_sequences([new_data])[0]]
    new_data = pad_sequences(new_data, maxlen=max_length)
    predictions = model.predict(new_data)

    # Convert the predictions to categorical
    predictions = np.argmax(predictions, axis=1)

    # Convert the predictions to sentiment labels
    sentiment_labels = ['Negative', 'Positive']
    predicted_sentiment = sentiment_labels[predictions[0]]

    # Print the predicted sentiment
    print('Predicted sentiment:', predicted_sentiment)

    return predicted_sentiment

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    review_text = request.form['review_text']
    predicted_sentiment = prediction(review)
    return render_template('index.html', prediction=predicted_sentiment, review_text=review)

if __name__ == '__main__':
    app.run(debug=True)