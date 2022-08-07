from flask import Flask, render_template, request, url_for
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
porter = PorterStemmer()

app = Flask(__name__)
bert_query = []
bert_docs = []
bert = []
tf_idf_query = []
tf_idf_docs = []
tf_idf_clean = []
tf_idf_final = []
tf_idf = []
bm25query = []
bm25corpus = []
bm25clean = []
bm25final = []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/bm25/', methods=('GET', 'POST'))
def bm25_f():
    if request.method == 'POST':
        bm25_query = request.form['query_bm25']
        bm25query.append(bm25_query)
        for x in range(21, 25):
            user_doc = request.form['doc' + str(x)]
            bm25corpus.append(user_doc)
        pre_list = bm25query + bm25corpus

        for _ in pre_list:
            a = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", _)
            stemmed_string_list = []
            a = a.split()
            for word in a:
                stemmed_string_list.append(porter.stem(word))
            a = " ".join(stemmed_string_list)
            bm25clean.append(a)
        tokenized_corpus = [doc.split(" ") for doc in bm25clean[1:]]
        bm25e = BM25Okapi(tokenized_corpus)
        tokenized_query = bm25clean[0].split(" ")

        doc_scores = bm25e.get_scores(tokenized_query).tolist()
        bert_score_pairs=list(zip(bm25corpus,doc_scores))
        bert_score_pairs=sorted(bert_score_pairs,key=lambda x:x[1],reverse=True)
        for doc,scores in bert_score_pairs:
            bm25final.append(doc)
        # tokens = []
        # for _ in range(len(doc_scores)):
        #     tokens.append(doc_scores.index(max(doc_scores)))
        #     doc_scores[doc_scores.index(max(doc_scores))] = doc_scores[doc_scores.index((max(doc_scores)))] * -1
        # # bm25clean.pop(0)
        #
        # for item in tokens:
        #     bm25final.append(bm25corpus[item])
    return render_template('bm25.html', bmlist=bm25final)


@app.route('/tfidf/', methods=('GET', 'POST'))
def tfidf_f():
    if request.method == 'POST':
        tfidf_query = request.form['query_tfidf']
        tf_idf_query.append(tfidf_query)
        for x in range(11, 15):
            user_doc = request.form['doc' + str(x)]
            tf_idf_docs.append(user_doc)
        pre_list = tf_idf_query + tf_idf_docs

        for _ in pre_list:
            a = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", _)
            stemmed_string_list = []
            a = a.split()
            for word in a:
                stemmed_string_list.append(porter.stem(word))
            a = " ".join(stemmed_string_list)
            tf_idf_clean.append(a)
        vectorizer = TfidfVectorizer(stop_words='english', strip_accents='unicode')
        a = vectorizer.fit_transform(tf_idf_clean)
        cosine_sim = cosine_similarity(a[0:1], a)
        u = sorted(cosine_sim.flatten().tolist(), reverse=True)
        # t = (cosine_sim.flatten().argsort()[::-1]).tolist()
        score_pairs = list(zip(tf_idf_docs, u))
        score_pairs = sorted(score_pairs, key=lambda k: k[1], reverse=True)
        for doc, scores in score_pairs:
            tf_idf_final.append(doc)
        # for y in t:
        #     tf_idf_final.append(pre_list[y])
        # tf_idf_final.pop(0)

    return render_template('tf-idf.html', finallist=tf_idf_final)


@app.route('/bert/', methods=('GET', 'POST'))
def bert_f():
    if request.method == 'POST':
        user_query = request.form['query']
        bert_query.append(user_query)
        for _ in range(4):
            user_doc = request.form['doc' + str(_)]
            bert_docs.append(user_doc)
        # Encode query and documents
        query_emb = model.encode(bert_query)
        doc_emb = model.encode(bert_docs)
        # Compute dot score between query and all document embeddings
        scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
        # Combine docs & scores
        doc_score_pairs = list(zip(bert_docs, scores))
        # Sort by decreasing score
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        for doc, scores in doc_score_pairs:
            bert.append(doc)

    return render_template('bert.html', bert=bert)


if __name__ == '__main__':
    app.run()
