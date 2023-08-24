from django.shortcuts import render
from django.http import JsonResponse
# import stanfordnlp
import spacy
from django.views.decorators.csrf import csrf_exempt


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
from django.http import JsonResponse

responses = {
    "привет": "Привет! Чем могу помочь?",
    "как дела?": "У меня всё отлично, спасибо!",
    "пока": "До свидания! Если у вас будут еще вопросы, обращайтесь.",
}

@csrf_exempt
def chat_view(request):
    if request.method == 'POST':
        user_message = request.POST.get("message", "").lower()
        bot_response = responses.get(user_message, "Извините, я не могу понять ваш вопрос.")

        response_data = {
            "user_message": user_message,
            "bot_response": bot_response,
        }

        return JsonResponse(response_data)








