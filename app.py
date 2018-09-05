from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
import pickle
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    #load model
    clf = joblib.load(open('nb_model.sav', 'rb'))
    train_labels = pickle.load(open('train_labels.pkl', 'rb'))
    train_X = pickle.load(open('train_descriptions.pkl', 'rb'))

    if request.method == 'POST':

        comment = request.form['comment']
        tfidf = TfidfVectorizer(stop_words='english', tokenizer=LemmaTokenizer())

        tfidf.fit(train_X, train_labels)

        data = tfidf.transform([comment]) #[article for article in comment]

        my_prediction = clf.predict(data)

    return render_template('results.html', prediction=my_prediction)

# def train_model():
#     divcorpus = pd.read_csv('diversity_corpus.csv')
#     masc_bow = divcorpus['genderSpecific']
#     neutral_bow = divcorpus['neutralEquality']
#     # format bow to list
#     masc_bow = masc_bow[:-2]
#     masc_bow = [i.lower() for i in list(masc_bow)]
#     neutral_bow = [i.lower() for i in list(neutral_bow)]
#
#     wordnet = WordNetLemmatizer()
#
#     jds = pd.read_csv('jds.csv')
#
#     # create article matrix
#     equality_mat = pd.DataFrame(columns =['articleID', 'num_neutrWords', 'num_specWords', 'neutroSpecRatio', 'score', 'label', 'neutrWords', 'specWords'])
#
#     corpus = jds['description']
#
#     # identify freq of terms that are masc and fem
#     lt = LemmaTokenizer()
#
#     for i, article in enumerate(corpus):
#         article_bow = lt.__call__(article)
#         count_n = 0
#         count_m = 0
#         neutrWords = []
#         specWords = []
#     for j in article_bow:
#         if j in neutral_bow:
#             count_n += 1
#             neutrWords.append(j)
#         if j in masc_bow:
#             count_m += 1
#             specWords.append(j)
#     df = pd.DataFrame({'articleID': [int(i)],
#               'num_neutrWords': [count_n],
#               'num_specWords': [count_m],
#               'neutrSpecRatio': [(1 + count_n)/(1 + count_m)],
#               'score': [(1 + count_n)/(1 + count_m) - 1],
#               'label': 0,
#               'neutrWords': [neutrWords],
#               'specWords': [specWords]} # = equal/neutral
#              )
#
#     equality_mat = pd.concat([equality_mat, df])
#     #equality_mat.astype({'articleID': 'int'})
#     #equality_mat.set_index('articleID', inplace=True)
#
#     equality_mat['label'][equality_mat['score'] < 0] = 1
#
#     labels = pd.DataFrame(data=equality_mat['label'])
#     labels.reset_index(inplace=True)
#     labels.drop('index', axis=1, inplace=True)
#     labels = np.array(labels).ravel()
#     labels = labels.astype('int')
#
#     #Create train, test labels
#     labels_train = labels[:100]
#     labels_test = list(labels[100:])
#
#     X = np.array(jds['description'][:100])
#
#     # Change job descriptions to bag of words
#     tfidf = TfidfVectorizer(stop_words='english', tokenizer=LemmaTokenizer())
#     tfidf_mat = tfidf.fit_transform(X, labels_train)
#
#     # Predict whether article is masc or neutral
#     # Naive Bayes
#     clf = BernoulliNB().fit(tfidf_mat.todense(), labels_train)
#     return clf

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.tokens = None
    def __call__(self, articles):
        self.tokens = [self.wnl.lemmatize(t) for t in word_tokenize(articles)]
        self.tokens = [i.lower() for i in self.tokens if i not in string.punctuation]
        self.tokens = [i for i in self.tokens if i not in string.ascii_letters]
        return self.tokens

if __name__ == '__main__':
    app.run(debug=True)
