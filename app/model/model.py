import pickle
import re
from pathlib import Path
import nltk
stemmer = nltk. SnowballStemmer("english")
import string
nltk.download('stopwords')
from nltk. corpus import stopwords
stopword=set(stopwords.words('english'))

from sklearn. feature_extraction. text import CountVectorizer
cv = CountVectorizer()

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)


classes = [
    0,1,2
]


def predict_pipeline(text):
    text = str (text). lower()
    text = re. sub('[.?]', '', text) 
    text = re. sub('https?://\S+|www.\S+', '', text)
    text = re. sub('<.?>+', '', text)
    text = re. sub('[%s]' % re. escape(string. punctuation), '', text)
    text = re. sub('\n', '', text)
    text = re. sub('\w\d\w', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ". join(text)
    text = [stemmer. stem(word) for word in text. split(' ')]
    text=" ". join(text)
    text = cv.transform([text]).toarray()
    pred = model.predict([text])
    return classes[pred[0]]