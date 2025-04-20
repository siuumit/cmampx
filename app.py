from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Ensure that stopwords are available
nltk.download('stopwords')

# Load stopwords
stopwords_set = set(stopwords.words('english'))
emoticon_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')

app = Flask(__name__)

# Load the sentiment analysis model and TF-IDF vectorizer
model_path = 'C:\\Users\\SUMIT\\Downloads\\Sentiment-Analysis-Mahcine-Learning-NLP-Project-main\\Sentiment-Analysis-Mahcine-Learning-NLP-Project-main\\clf.pkl'
vectorizer_path = 'C:\\Users\\SUMIT\\Downloads\\Sentiment-Analysis-Mahcine-Learning-NLP-Project-main\\Sentiment-Analysis-Mahcine-Learning-NLP-Project-main\\tfidf.pkl'

with open(model_path, 'rb') as f:
    clf = pickle.load(f)
with open(vectorizer_path, 'rb') as f:
    tfidf = pickle.load(f)

def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoticon_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(text)

@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    if request.method == 'POST':
        comment = request.form.get('comment')

        # Preprocess the comment
        preprocessed_comment = preprocessing(comment)

        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comment])

        # Predict the sentiment
        sentiment = clf.predict(comment_vector)[0]

        return render_template('index.html', sentiment=sentiment)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
