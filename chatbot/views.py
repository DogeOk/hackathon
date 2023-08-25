from django.shortcuts import render
from django.http import JsonResponse
# import stanfordnlp
import spacy
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.conf import settings
import os

# stanfordnlp.download("ru")
# # StanfordNLP
# nlp = stanfordnlp.Pipeline(lang="ru", processors="tokenize")
# nlp = spacy.load('ru_core_news_sm')

# responses = {
#     "привет": "Привет! Чем могу помочь?",
#     "как дела?": "У меня всё отлично, спасибо!",
#     "пока": "До свидания! Если у вас будут еще вопросы, обращайтесь.",
# }
def chat_interface(request):
    return render(request, 'chat.html')

# @csrf_exempt
# def chat_view(request):
#     if (request.headers.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest' 
#         and request.method == 'POST'):
#         user_message = request.POST.get("message", "").lower()
#         doc = nlp(user_message)
#         keywords = [token.text for token in doc if token.is_alpha]
        
#         bot_response = "Извините, я не могу понять ваш вопрос."
#         # for sentence in doc.sentences:
#         #     for word in sentence.words:
#         #         if word.lemma.lower() in responses:
#         #             bot_response = responses[word.lemma.lower()]
#         #             break
#         for keyword in keywords:
#             if keyword in responses:
#                 bot_response = responses[keyword]
#                 break
#         response_data = {
#             "user_message": user_message,
#             "bot_response": bot_response,
#         }

#         return JsonResponse(response_data)
#     else:
#         return JsonResponse({"message": "hello"})
# from django.http import JsonResponse

# responses = {
#     "привет": "Привет! Чем могу помочь?",
#     "как дела?": "У меня всё отлично, спасибо!",
#     "пока": "До свидания! Если у вас будут еще вопросы, обращайтесь.",
# }

@csrf_exempt
# def chat_view(request):
#     if request.method == 'POST':
#         user_message = request.POST.get("message", "").lower()
#         bot_response = responses.get(user_message, "Извините, я не могу понять ваш вопрос.")

#         response_data = {
#             "user_message": user_message,
#             "bot_response": bot_response,
#         }

#         return HttpResponse(response_data["bot_response"])


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
    max_similarity = 0.0
    nlp = spacy.load("ru_core_news_sm")
    for index, row in faq_data.iterrows():
        question = row['Вопрос'].lower()
        response = row['Ответ']

        doc_user = nlp(user_message)
        doc_question = nlp(question)

        similarity = doc_user.similarity(doc_question)

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = response

    if max_similarity < 0.5:
        return "Извините, я не могу понять ваш вопрос."

    return best_match












