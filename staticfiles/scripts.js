// scripts.js
$(document).ready(function() {
    // Открытие окна чата при нажатии на кнопку
    $('#chat-button').click(function() {
        $('#chat-container').show();
    });

    // Закрытие окна чата при нажатии на кнопку "Закрыть"
    $('#close-button').click(function() {
        $('#chat-container').hide();
    });

    // Отправка сообщения на сервер при нажатии на кнопку "Отправить"
    $('#send-button').click(function() {
        var userMessage = $('#message-input').val();
        $('#message-input').val(''); // Очистка поля ввода

        // Получение CSRF токена
        var csrftoken = $('[name=csrfmiddlewaretoken]').val();

        // Отправка AJAX-запроса на сервер
        $.ajax({
            url: '/chat/', 
            type: 'POST',
            data: {
                message: userMessage,
                csrfmiddlewaretoken: csrftoken, // Включение CSRF токена в данные запроса
            },
            success: function(data) {
                // Отображение ответа от сервера
                $('#chat-messages').append('<p><strong>Вы:</strong> ' + userMessage + '</p>');
                $('#chat-messages').append('<p><strong>Бот:</strong> ' + data.bot_response + '</p>');
            }
        });
    });
});





