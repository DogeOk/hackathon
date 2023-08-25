from django.shortcuts import render
from django.http import JsonResponse
import spacy
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.conf import settings
from llama_index import ServiceContext, LLMPredictor, PromptHelper
from llama_index.llms import LlamaCPP
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
import os
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import GPTVectorStoreIndex, ServiceContext, download_loader, load_index_from_storage
def chat_interface(request):
    return render(request, 'chat.html')



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



@csrf_exempt
def chat_view(request):
    if request.method == 'POST':
        user_message = request.POST.get("message", "").lower()

        faq_file_path = os.path.join(settings.BASE_DIR, 'faq.xlsx')
        faq_data = pd.read_excel(faq_file_path, engine='openpyxl')
        
        bot_response = find_bot_response(user_message, faq_data)

        response_data = {
            "user_message": user_message,
            "bot_response": bot_response,
        }

        return HttpResponse(response_data["bot_response"])

def find_bot_response(user_message, faq_data):
    best_match = None
    min_distance = float('inf')

    for index, row in faq_data.iterrows():
        question = row['QUESTION'].lower()
        response = row['ANSWER']

        distance = levenshtein(user_message, question)

        if distance < min_distance:
            min_distance = distance
            best_match = response

    if min_distance > len(user_message) / 2:
        return "Извините, я не могу понять ваш вопрос."

    return best_match

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]















