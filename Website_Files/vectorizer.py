# Stand Alone Hashing Vector File
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import os

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(
                os.path.join(cur_dir, 
                'pkl_objects', 
                'stopwords.pkl'), 'rb'))

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

def tokenizer_porter(text):
    porter = PorterStemmer()
    line = [porter.stem(word) for word in text.split()]
    return line

vect_optimized = HashingVectorizer(decode_error='ignore',
                         norm = None,
                         n_features=2**21,
                         preprocessor=preprocessor,
                         stop_words = stop,
                         tokenizer=tokenizer_porter)


