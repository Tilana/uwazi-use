# export TFHUB_CACHE_DIR=./tfhub_modules
# export FLASK_APP=use_routes.py
# flask run --port 4000

from flask import Flask, request
import pandas as pd
import json
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

app = Flask(__name__)
model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
THRESHOLD = 0.65
SENTENCE_LENGTH = 30
TRAINING_DOC = 'use_training.csv'

def prepro(sentence):
    result = sentence.replace(",", " , ")
    result = result.replace(":", " : ")
    result = result.replace(";", " ; ")
    return result;

def get_sentence_length(text):
    return len(text.split(' '))

def split_into_sentences(text, size):
    words = text.split()
    arrs = []
    while len(words) > size:
        piece = words[:size]
        arrs.append(piece)
        words = words[size:]
    arrs.append(words)
    sentences = []
    for one_sentence in arrs:
        one_sentence = ' '.join(one_sentence)
        sentences.append(one_sentence)
    return sentences

def get_similar_sentences(similarity, evidences, sentences, doc_id):
    similar_sentences = pd.DataFrame(columns=['probability'])
    for ind, sentence in enumerate(sentences):
        for pos,sim in enumerate(similarity[:,ind]):
            if sim>=THRESHOLD:
                similar_sentences = similar_sentences.append({'evidence':sentence, 'probability':sim, 'label':1, 'document':doc_id, 'property':evidences.loc[pos]['property'], 'value':evidences.loc[pos]['value']}, ignore_index=True)
    return similar_sentences


@app.route('/classification/train', methods=['POST'])
def train_route():
    data = json.loads(request.data)
    sentence = data['evidence']['text'].encode('utf-8');
    sentence = prepro(sentence)

    if data['isEvidence']:
        df = pd.DataFrame({'sentence': sentence, 'property':data['property'], 'value':data['value']}, index=[0])
        if os.path.exists(TRAINING_DOC):
            df.to_csv(TRAINING_DOC, mode='a', header=False, index=False, encoding='utf8')
        else:
            df.to_csv(TRAINING_DOC, mode='a', index=False, encoding='utf8')
    return "{}"


@app.route('/classification/retrain', methods=['POST'])
def retrain_route():
    return "{}"


@app.route('/classification/predictOneModel', methods=['POST'])
def predict_one_model():
    data = json.loads(request.data)
    docs = pd.read_json(json.dumps(data['docs']), encoding='utf-8')

    evidences = pd.read_csv(TRAINING_DOC, encoding='utf8')
    evidences = evidences[(evidences['property'] == data['property']) & (evidences['value'] == data['value'])]
    evidences = evidences.reset_index()
    evidences['sentence_length'] = evidences.sentence.apply(get_sentence_length)
    average_sentence_length = evidences['sentence_length'].mean()

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        evidence_embedding = session.run(model(evidences['sentence'].tolist()))
        similar_sentences = pd.DataFrame(columns=['probability'])
        for doc in docs.iterrows():
            sentences = split_into_sentences(doc[1].text, int(average_sentence_length))
            sentence_embedding = session.run(model(sentences))

            similarity = np.matmul(evidence_embedding, np.transpose(sentence_embedding))
            similar_sentences = similar_sentences.append(get_similar_sentences(similarity, evidences, sentences, doc[1]['_id']))

        similar_sentences.sort_values(by=['probability'], ascending=False, inplace=True)
        similar_sentences.drop_duplicates(inplace=True)

    return similar_sentences.to_json(orient='records')


@app.route('/classification/predict', methods=['POST'])
def predict_route():
    data = json.loads(request.data)
    doc = pd.read_json('[' + json.dumps(data['doc']) + ']', encoding='utf8').loc[0];
    doc_id = data['properties'][0]['document']

    sentences = split_into_sentences(doc['text'], SENTENCE_LENGTH)
    evidences = pd.read_csv(TRAINING_DOC, encoding='utf8')

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        evidence_embedding = session.run(model(evidences['sentence'].tolist()))
        sentence_embedding = session.run(model(sentences))

        similarity = np.matmul(evidence_embedding, np.transpose(sentence_embedding))
        similar_sentences = get_similar_sentences(similarity, evidences, sentences, doc_id)

        similar_sentences.sort_values(by=['probability'], ascending=False, inplace=True)
        similar_sentences.drop_duplicates(inplace=True)

    return similar_sentences.to_json(orient='records')
