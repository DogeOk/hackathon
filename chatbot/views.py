from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from joblib import load
import os
import re
import numpy as np
import torch
import pandas as pd
import spacy
import transformers
import snoop

toxic_answer = (f"Ай-яй-яй, как некрасиво! Вы же в культурной столице! " 
                f"Введите, пожалуйста, корректный запрос 🤓")
not_found_answer = (f"Как истинный лев, я прошерстил всю библиотеку, но " 
                    f"так и смог найти подходящий ответ. Смилуйтесь, " 
                    f"уточните Ваш запрос! 🥺")


def chat_interface(request):
    return render(request, 'chat.html')


@csrf_exempt
def chat_view(request):
    user_message = request.POST.get("message", "").lower()

    if toxic_predict(user_message):
        return HttpResponse(toxic_answer)
    
    # if action == "process_response":

    #     return process_response(user_message)


    faq_file_path = os.path.join(settings.BASE_DIR, 'full_data.csv')
    faq_data = pd.read_csv(faq_file_path)

    # if any(char.isdigit() for char in user_message):
    #     return HttpResponse(number_questions(user_message, faq_data))
    # if action == "clarification":
    #     return handle_clarification(user_message, faq_data)
    # else:
    #     return handle_regular_question(user_message, faq_data)
    bot_response = find_bot_response(user_message, faq_data)
    # global levenshtein_a
    # levenshtein_a, levenshtein_q = find_levenshtein_match(user_message, faq_data)

    response_data = {
        "user_message": user_message,
        "bot_response": bot_response,  
    }

    return HttpResponse(response_data["bot_response"])
    
# def process_response(client_response):
#     # Здесь обрабатываем ответ клиента
#     if client_response == "да":
#         # Обработка для "Да"
#         bot_response = levenshtein_a
#     elif client_response == "нет":
#         # Обработка для "Нет"
#         # Продолжаем с find_tfidf_match
#         tfidf_a, tfidf_q = find_tfidf_match(user_message, faq_data)
#         bot_response = tfidf_q

#     response_data = {
#         "user_message": client_response,
#         "bot_response": bot_response,
#         "action": "process_response"  # Указываем action для обработки следующего ответа клиента
#     }

#     return JsonResponse({"response": response_data})
    
# Question with number
def number_questions(user_message, faq_data):

    pattern = r'услуг[аеу]?\s+(\d+)'

    match = re.search(pattern, user_message)
    service_number = match.group(1)
    
    matching_questions = {}
    for index, row in faq_data.iterrows():
        question = row['question']
        if str(service_number) in question.lower():
            matching_questions[index] = question

    try:
        best_match_index = find_best_matching_question(user_message, 
                                                   matching_questions)
        best_match = faq_data.loc[best_match_index, 'answer']
        
        return best_match
    except:
        return not_found_answer

def find_best_matching_question(user_message, matching_questions):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(matching_questions.values()) 
    user_message_tfidf = tfidf_vectorizer.transform([user_message])

    cosine_similarities = cosine_similarity(user_message_tfidf, tfidf_matrix)
    best_match_index = cosine_similarities.argmax()
    best_match_question_index = list(matching_questions.keys())[best_match_index]

    return best_match_question_index



# Levenshtein algothim
def find_levenshtein_match(user_message, faq_data):
    best_answer = None
    max_similarity = 0

    for question in faq_data['question']:
        similarity = fuzz.ratio(user_message, question)
        if similarity > max_similarity:
            max_similarity = similarity
            best_question = question
        
    best_answer = faq_data.loc[faq_data['question'] == best_question]['answer']\
                .values[0]

    return best_answer, best_question

# TF-IDF algothim
def find_tfidf_match(user_message, faq_data):
    tfidf_vectorizer = TfidfVectorizer()
    faq_questions = faq_data['question']
    tfidf_matrix = tfidf_vectorizer.fit_transform(faq_questions)

    user_message_tfidf = tfidf_vectorizer.transform([user_message])

    cosine_similarities = cosine_similarity(user_message_tfidf, tfidf_matrix)

    best_match_index = cosine_similarities.argmax()
    best_answer = faq_data['answer'].iloc[best_match_index]
    best_question = faq_data['question'].iloc[best_match_index]

    return best_answer, best_question


@snoop
def find_bot_response(user_message, faq_data):
  
    # Левенштейн
    levenshtein_a, levenshtein_q = find_levenshtein_match(user_message, faq_data) 

    # TF-IDF
    tfidf_match_a, tfidf_match_q = find_tfidf_match(user_message, faq_data)
    
    #LangChain
    loaded_svm_retriever = load(os.path.join(settings.BASE_DIR, 
                                            "svm_retriever.pkl"))
    docs_svm=loaded_svm_retriever.get_relevant_documents(user_message)

    try:
        return(levenshtein_a)
    except IndexError:
        try:
            return(tfidf_match_a)
        except:
            try:
                for i in range(len(docs_svm)):
                    try:
                        return docs_svm[i].metadata["answer"]
                    except:
                        pass
            except:
                return(not_found_answer)

   

def toxic_predict(message):
    PATH = 's-nlp/russian_toxicity_classifier'
    tokenizer = transformers.BertTokenizer.from_pretrained(PATH)
    message = tokenizer.encode(message, add_special_tokens=True, 
                               max_length=512, 
                               truncation=True)
    max_len = 512
    padded = np.array([message+[0]*(max_len - len(message))])
    attention_mask = np.where(padded != 0, 1, 0)
    model = transformers.BertModel.from_pretrained(PATH)
    batch_size = 1
    embeddings = []
    for i in range(padded.shape[0] // batch_size):
            batch = torch.LongTensor(padded[batch_size*i:batch_size*(i+1)])
            attention_mask_batch = torch.LongTensor(
                attention_mask[batch_size*i:batch_size*(i+1)]
                )

            with torch.no_grad():
                batch_embeddings = model(batch, 
                                         attention_mask=attention_mask_batch
                                         )

            embeddings.append(batch_embeddings[0][:,0,:].numpy())
    return (load('./models/toxic_model.joblib')\
            .predict(np.concatenate(embeddings))[0])