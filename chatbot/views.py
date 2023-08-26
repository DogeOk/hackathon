from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.conf import settings
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from pymystem3 import Mystem
from joblib import dump, load
import os
import re
import numpy as np
import torch
import pandas as pd
import spacy
import transformers


def chat_interface(request):
    return render(request, 'chat.html')


@csrf_exempt
def chat_view(request):
    if request.method == 'POST':
        user_message = request.POST.get("message", "").lower()

        if toxic_predict(user_message):
            return HttpResponse("Ай-яй-яй, как некрасиво! Вы же в культурной столице! Введите, пожалуйста, корректный запрос 🤓")

        faq_file_path = os.path.join(settings.BASE_DIR, 'full_data.csv')
        faq_data = pd.read_csv(faq_file_path)

        if any(char.isdigit() for char in user_message):
            return HttpResponse(number_questions(user_message, faq_data))
        
        bot_response = find_bot_response(user_message, faq_data)

        response_data = {
            "user_message": user_message,
            "bot_response": bot_response,
        }

        return HttpResponse(response_data["bot_response"])
    

def number_questions(user_message, faq_data):

    pattern = r'услуг[аеу]?\s+(\d+)'

    match = re.search(pattern, user_message)
    service_number = int(match.group(1))
    
    matching_questions = {}
    for index, row in faq_data.iterrows():
        question = row['question']
        if str(service_number) in question.lower():
            matching_questions[index] = question


    best_match_index = find_best_matching_question(user_message, 
                                                   matching_questions)
    
    best_match = faq_data.loc[best_match_index, 'answer']

    return best_match

def find_best_matching_question(user_message, matching_questions):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(matching_questions.values()) 
    user_message_tfidf = tfidf_vectorizer.transform([user_message])

    cosine_similarities = cosine_similarity(user_message_tfidf, tfidf_matrix)
    best_match_index = cosine_similarities.argmax()
    best_match_question_index = list(matching_questions.keys())[best_match_index]

    return best_match_question_index



def create_word2vec_model(faq_data):
    sentences = [text.split() for text in faq_data['question']]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
    return model


def find_word2vec_match(user_message, faq_data):
    user_tokens = user_message.split()
    similarity = -1
    best_match = None
    model = create_word2vec_model(faq_data)

    for question in model.wv.index_to_key:
        question_tokens = question.split()
        sim = model.wv.n_similarity(user_tokens, question_tokens)
        if sim > similarity:
            similarity = sim
            best_match = question

    return best_match


def find_levenshtein_match(user_message, faq_data):
    best_match = None
    max_similarity = 0

    for question in faq_data['question']:
        similarity = fuzz.ratio(user_message, question)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = question

    return best_match


def find_tfidf_match(user_message, faq_data):
    tfidf_vectorizer = TfidfVectorizer()
    faq_questions = faq_data['question']
    tfidf_matrix = tfidf_vectorizer.fit_transform(faq_questions)

    user_message_tfidf = tfidf_vectorizer.transform([user_message])

    cosine_similarities = cosine_similarity(user_message_tfidf, tfidf_matrix)

    best_match_index = cosine_similarities.argmax()
    best_match_response = faq_data['answer'].iloc[best_match_index]

    return best_match_response


def find_spacy_match(user_message, faq_data):
    best_match = None
    max_similarity = 0.0
    nlp = spacy.load("ru_core_news_sm")
    for index, row in faq_data.iterrows():
        question = row['question'].lower()
        response = row['answer']

        doc_user = nlp(user_message)
        doc_question = nlp(question)

        similarity = doc_user.similarity(doc_question)

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = response

    return best_match


def find_bot_response(user_message, faq_data):

    # Word2Vec
    word2vec_match = find_word2vec_match(user_message, faq_data)
    max_similarity = 0.7
    
    # Левенштейн
    levenshtein_match = find_levenshtein_match(user_message, faq_data)
    levenshtein_similarity = fuzz.ratio(user_message, levenshtein_match) / 100  
    
    # TF-IDF
    tfidf_match = find_tfidf_match(user_message, faq_data)

    #SpaCy
    spacy_match = find_tfidf_match(user_message, faq_data)

    best_match = None
    
    if levenshtein_similarity > max_similarity:
        best_match = levenshtein_match
    elif word2vec_match:
        best_match = word2vec_match
    elif tfidf_match:
        best_match = tfidf_match
    else:
        best_match = spacy_match   
    
    if not best_match:
        return "Извините, я не могу понять ваш вопрос."

    return faq_data.loc[faq_data['question'] == best_match]['answer'].values[0]
    

def toxic_predict(message):
    PATH = 's-nlp/russian_toxicity_classifier'
    tokenizer = transformers.BertTokenizer.from_pretrained(PATH)
    message = tokenizer.encode(message, add_special_tokens=True, max_length=512, truncation=True)
    max_len = 512
    padded = np.array([message+[0]*(max_len - len(message))])
    attention_mask = np.where(padded != 0, 1, 0)
    model = transformers.BertModel.from_pretrained(PATH)
    batch_size = 1
    embeddings = []
    for i in range(padded.shape[0] // batch_size):
            batch = torch.LongTensor(padded[batch_size*i:batch_size*(i+1)])
            attention_mask_batch = torch.LongTensor(attention_mask[batch_size*i:batch_size*(i+1)])

            with torch.no_grad():
                batch_embeddings = model(batch, attention_mask=attention_mask_batch)

            embeddings.append(batch_embeddings[0][:,0,:].numpy())
    return load('./models/toxic_model.joblib').predict(np.concatenate(embeddings))[0]