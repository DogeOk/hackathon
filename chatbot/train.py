import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from fuzzywuzzy import fuzz
from elasticsearch import Elasticsearch

# Загрузка данных FAQ
def load_faq_data():
    faq_data = pd.read_csv("/Users/nikolajnecaev/Desktop/YANDEX/portfolio/chatbot_project/full_data.csv")
    return faq_data

# Создание индекса для TF-IDF
def create_tfidf_index(faq_data):
    tfidf_vectorizer = TfidfVectorizer()
    faq_texts = faq_data['question'] + " " + faq_data['answer']
    tfidf_matrix = tfidf_vectorizer.fit_transform(faq_texts)
    return tfidf_matrix

# Создание модели Word2Vec
def create_word2vec_model(faq_data):
    sentences = [text.split() for text in faq_data['question']]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
    return model

# Функция для нахождения ближайшего вектора Word2Vec
def find_word2vec_match(user_message, model):
    user_tokens = user_message.split()
    similarity = -1
    best_match = None

    for question in model.wv.index_to_key:
        question_tokens = question.split()
        sim = model.wv.n_similarity(user_tokens, question_tokens)
        if sim > similarity:
            similarity = sim
            best_match = question

    return best_match

# Функция для нахождения наиболее похожего вопроса с использованием расстояния Левенштейна
def find_levenshtein_match(user_message, faq_data):
    best_match = None
    max_similarity = 0

    for question in faq_data['question']:
        similarity = fuzz.ratio(user_message, question)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = question

    return best_match

# Функция для поиска наиболее похожего ответа с использованием TF-IDF
def find_tfidf_match(user_message, faq_data, tfidf_matrix):
    tfidf_vectorizer = TfidfVectorizer()
    user_message_tfidf = tfidf_vectorizer.transform([user_message])

    cosine_similarities = cosine_similarity(user_message_tfidf, tfidf_matrix)

    best_match_index = cosine_similarities.argmax()
    best_match_response = faq_data['answer'].iloc[best_match_index]

    return best_match_response

# Функция для поиска наиболее похожего ответа в Elasticsearch
def find_elasticsearch_match(user_message, es_client, index_name):
    search_body = {
        "query": {
            "match": {
                "question": user_message
            }
        }
    }

    results = es_client.search(index=index_name, body=search_body)
    if results['hits']['total']['value'] > 0:
        best_match_response = results['hits']['hits'][0]['_source']['answer']
        return best_match_response
    else:
        return None

# Инициализация Elasticsearch клиента
es_client = Elasticsearch(["http://localhost:9200"])


# Загрузка данных FAQ
faq_data = load_faq_data()

# Создание индексов для TF-IDF и Word2Vec
tfidf_matrix = create_tfidf_index(faq_data)
word2vec_model = create_word2vec_model(faq_data)

# Elasticsearch индексирование
index_name = "art"
es_client.indices.create(index=index_name, ignore=400)
for _, row in faq_data.iterrows():
    document = {
        "question": row['question'],
        "answer": row['answer']
    }
    es_client.index(index=index_name, body=document)

# Пример использования всех методов
user_message = "как получить выплаты по детской карте"
word2vec_match = find_word2vec_match(user_message, word2vec_model)
levenshtein_match = find_levenshtein_match(user_message, faq_data)
tfidf_match = find_tfidf_match(user_message, faq_data, tfidf_matrix)
elasticsearch_match = find_elasticsearch_match(user_message, es_client, index_name)

print("Word2Vec Match:", word2vec_match)
print("Levenshtein Match:", levenshtein_match)
print("TF-IDF Match:", tfidf_match)
print("Elasticsearch Match:", elasticsearch_match)