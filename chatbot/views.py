from django.shortcuts import render
import spacy
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.conf import settings
import os


def chat_interface(request):
    return render(request, 'chat.html')
from transformers import BertTokenizer, BertForQuestionAnswering
import torch


from fuzzywuzzy import fuzz
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
from elasticsearch import Elasticsearch
import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
from gensim.models import Word2Vec


# def chat_view(request):
#     if request.method == 'POST':
#         user_message = request.POST.get("message", "").lower()
#         bot_response = responses.get(user_message, "Извините, я не могу понять ваш вопрос.")

#         response_data = {
#             "user_message": user_message,
#             "bot_response": bot_response,
#         }

#         return HttpResponse(response_data["bot_response"])




# def chat_view(request):
#     if request.method == 'POST':
#         user_message = request.POST.get("message", "").lower()

#         faq_file_path = os.path.join(settings.BASE_DIR, 'faq.xlsx')
#         faq_data = pd.read_excel(faq_file_path, engine='openpyxl')
        
#         bot_response = find_bot_response(user_message, faq_data)

#         response_data = {
#             "user_message": user_message,
#             "bot_response": bot_response,
#         }

#         return HttpResponse(response_data["bot_response"])

# def find_bot_response(user_message, faq_data):
#     best_match = None
#     max_similarity = 0.0
#     nlp = spacy.load("ru_core_news_sm")
#     for index, row in faq_data.iterrows():
#         question = row['QUESTION'].lower()
#         response = row['ANSWER']

#         doc_user = nlp(user_message)
#         doc_question = nlp(question)

#         similarity = doc_user.similarity(doc_question)

#         if similarity > max_similarity:
#             max_similarity = similarity
#             best_match = response

#     if max_similarity < 0.5:
#         return "Извините, я не могу понять ваш вопрос."

#     return best_match






# def chat_view(request):
#     if request.method == 'POST':
#         user_message = request.POST.get("message", "").lower()

#         faq_file_path = os.path.join(settings.BASE_DIR, 'faq.xlsx')
#         faq_data = pd.read_excel(faq_file_path, engine='openpyxl')
        
#         bot_response = find_bot_response(user_message, faq_data)

#         response_data = {
#             "user_message": user_message,
#             "bot_response": bot_response,
#         }

#         return HttpResponse(response_data["bot_response"])

# def find_bot_response(user_message, faq_data):
#     best_match = None
#     min_distance = float('inf')

#     for index, row in faq_data.iterrows():
#         question = row['QUESTION'].lower()
#         response = row['ANSWER']

#         distance = levenshtein(user_message, question)

#         if distance < min_distance:
#             min_distance = distance
#             best_match = response

#     if min_distance > len(user_message) / 2:
#         return "Извините, я не могу понять ваш вопрос."

#     return best_match

# def levenshtein(s1, s2):
#     if len(s1) < len(s2):
#         return levenshtein(s2, s1)

#     if len(s2) == 0:
#         return len(s1)

#     previous_row = range(len(s2) + 1)
#     for i, c1 in enumerate(s1):
#         current_row = [i + 1]
#         for j, c2 in enumerate(s2):
#             insertions = previous_row[j + 1] + 1
#             deletions = current_row[j] + 1
#             substitutions = previous_row[j] + (c1 != c2)
#             current_row.append(min(insertions, deletions, substitutions))
#         previous_row = current_row

#     return previous_row[-1]




# def load_faq_data():
#     faq_file_path = os.path.join(settings.BASE_DIR, 'faq.xlsx')
#     faq_data = pd.read_excel(faq_file_path, engine='openpyxl')
#     return faq_data

# def create_faq_index(faq_data):
#     index = {}
#     for _, row in faq_data.iterrows():
#         question = str(row['QUESTION'])
#         response = str(row['ANSWER'])
#         index[question] = response
#     return index

# def find_bot_response(user_message, faq_index, tokenizer, model):
#     best_match = None
#     max_similarity = 0.0

#     for question, response in faq_index.items():
#         inputs = tokenizer.encode_plus(user_message, question, 
#                                        return_tensors="pt", max_length=512, 
#                                        truncation=True)
#         input_ids = inputs["input_ids"]
#         attention_mask = inputs["attention_mask"]

#         outputs = model(input_ids, attention_mask=attention_mask)
#         start_logits = outputs.start_logits
#         end_logits = outputs.end_logits

#         answer_start = torch.argmax(start_logits)
#         answer_end = torch.argmax(end_logits)

#         answer = tokenizer.convert_tokens_to_string(
#             tokenizer.convert_ids_to_tokens(
#                 input_ids[0][answer_start:answer_end+1]
#             )
#         )

#         if answer:
#             similarity = len(answer) / len(user_message)
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 best_match = response

#     if max_similarity < 0.5:
#         print("Извините, я не могу понять ваш вопрос.")

#     return best_match


# @csrf_exempt
# def chat_view(request):
#     if request.method == 'POST':
#         user_message = request.POST.get("message", "")

#         faq_data = load_faq_data()
#         faq_index = create_faq_index(faq_data)

#         # BERT
#         model_name = "roberta-large"
#         tokenizer = RobertaTokenizer.from_pretrained(model_name)
#         model = RobertaForQuestionAnswering.from_pretrained(model_name)


#         bot_response = find_bot_response(user_message, faq_index, tokenizer, model)

#         response_data = {
#             "user_message": user_message,
#             "bot_response": bot_response,
#         }

#         return HttpResponse(response_data["bot_response"])

@csrf_exempt
def chat_view(request):
    if request.method == 'POST':
        user_message = request.POST.get("message", "").lower()

        faq_file_path = os.path.join(settings.BASE_DIR, 'full_data.csv')
        faq_data = pd.read_csv(faq_file_path)
        
        bot_response = find_bot_response(user_message, faq_data)

        response_data = {
            "user_message": user_message,
            "bot_response": bot_response,
        }

        return HttpResponse(response_data["bot_response"])



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
    # print(word2vec_match, levenshtein_match, find_tfidf_match)

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
    print(best_match)
    return faq_data.loc[faq_data['question'] == best_match]['answer'].values[0]














