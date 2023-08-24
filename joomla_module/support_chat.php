<?php defined('_JEXEC') or die;?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<style>
    .chat-window {
        width: 400px;
        box-shadow: 0px 0px 30px 9px rgba(0,0,0,0.5);
        -webkit-box-shadow: 0px 0px 30px 9px rgba(0,0,0,0.5);
        -moz-box-shadow: 0px 0px 30px 9px rgba(0,0,0,0.5);
        border-radius: 10px 10px 0 0;
        position: absolute;
        bottom: 0;
        right: 20vw;
        overflow-y: hidden;
        z-index: 1;
    }
    .chat-header {
        display: grid;
        grid-template-columns: 4fr 1fr;
        padding: 15px 15px;
        border-bottom: 2px solid gray;
        background: rgb(231,212,194);
        background: linear-gradient(45deg, rgba(231,212,194,1) 0%, rgba(206,165,129,1) 50%);
        border-radius: 10px 10px 0 0;
        cursor: pointer;
    }
    .chat-title {
        font-family: Tahoma, sans-serif;
        font-weight: bold;
        justify-self: left;
        cursor: default;
        cursor: pointer;
    }
    .chat-close-btn {
        font-family: Tahoma, sans-serif;
        justify-self: right;
        cursor: pointer;
        font-size: 18px;
    }
    .chat-history {
        display: grid;
        grid-template-rows: repeat(4, fit-content(50px));
        background-color: #ebedf0;
        height: 400px;
        overflow-y: scroll;
        align-items: flex-start;
    }
    .chat-history::-webkit-scrollbar {
        width: 12px;
    }
    .chat-history::-webkit-scrollbar-track {
        background-color: #ced1d6;
    }
    .chat-history::-webkit-scrollbar-thumb {
        background-color: gray;
        border-radius: 20px;
        border: 3px solid #ced1d6;
    }
    .avatar {
        width: 32px;
        height: 32px;
    }
    .chat-client-message {
        justify-self: right;
        display: grid;
        justify-items: right;
        align-items: flex-end;
        grid-template-columns: 1fr fit-content(32px);
        margin: 10px 10px;
        grid-gap: 10px;
    }
    .chat-client-message div {
        background: #cea581;
        padding: 15px;
        border-radius: 15px;
        max-width: 60%;
        word-break: break-word;
    }
    .chat-support-message {
        justify-self: left;
        display: grid;
        justify-items: left;
        align-items: flex-end;
        grid-template-columns: fit-content(32px) 1fr;
        margin: 10px 10px;
        grid-gap: 10px;
    }
    .chat-support-message div {
        background-color: #e7d4c2;
        padding: 15px;
        border-radius: 15px;
        max-width: 60%;
        word-break: break-word;
    }
    .chat-sender {
        display: grid;
        grid-template-columns: 4fr 1fr;
        border-top: 2px solid gray;
        grid-gap: 10px;
    }
    .chat-input {
        resize: none;
        border: 0px;
        padding: 10px;
        font-size: 20px;
        width: 100%;
    }
    .chat-input::-webkit-scrollbar {
        width: 0px;
    }
    .chat-send-btn {
        justify-self: center;
        align-self: center;
        cursor: pointer;
    }
</style>
<body>
    <div class="chat-window">
        <div class="chat-header">
            <div class="chat-title">Нужна помощь?</div>
            <div class="chat-close-btn"></div>
        </div>
        <div class="chat-history">
            <div class="chat-support-message">
                <img class="avatar" src="modules/support_chat/avatar.png" alt="avatar"/>
                <div>Здраствуйте! Меня зовут Ярослав. Буду рад избавить Вас от страданий и найти необходимую информацию! Пожалуйста, напишите название услуги или ведомства</div>
            </div>
        </div>
        <div class="chat-sender">
            <textarea class="chat-input" rows="2" placeholder="Введите запрос..."></textarea>
            <svg class="chat-send-btn" width="48px" height="48px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M11.5003 12H5.41872M5.24634 12.7972L4.24158 15.7986C3.69128 17.4424 3.41613 18.2643 3.61359 18.7704C3.78506 19.21 4.15335 19.5432 4.6078 19.6701C5.13111 19.8161 5.92151 19.4604 7.50231 18.7491L17.6367 14.1886C19.1797 13.4942 19.9512 13.1471 20.1896 12.6648C20.3968 12.2458 20.3968 11.7541 20.1896 11.3351C19.9512 10.8529 19.1797 10.5057 17.6367 9.81135L7.48483 5.24303C5.90879 4.53382 5.12078 4.17921 4.59799 4.32468C4.14397 4.45101 3.77572 4.78336 3.60365 5.22209C3.40551 5.72728 3.67772 6.54741 4.22215 8.18767L5.24829 11.2793C5.34179 11.561 5.38855 11.7019 5.407 11.8459C5.42338 11.9738 5.42321 12.1032 5.40651 12.231C5.38768 12.375 5.34057 12.5157 5.24634 12.7972Z" stroke="#000000" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
    </div>
    <script>
        chat_window = document.querySelector('.chat-window');
        chat_header = document.querySelector('.chat-header');
        chat_title = document.querySelector('.chat-title');
        chat_close_btn = document.querySelector('.chat-close-btn');
        chat_input = document.querySelector('.chat-input');
        chat_send_btn = document.querySelector('.chat-send-btn');
        const FULL_WINDOW = chat_window.offsetHeight;
        const CLOSE_WINDOW = chat_header.offsetHeight;
        chat_window.style.height = CLOSE_WINDOW + 'px';

        is_chat_open = false;
        is_close_btn_clicked = false;
        chat_header.addEventListener('click', function() {
            if (!is_chat_open) {
                for (let i = 0; i < 20; i++) {
                    setTimeout(function () {
                        chat_window.style.height = CLOSE_WINDOW + (FULL_WINDOW - CLOSE_WINDOW) / 20 * (i + 1) + 'px';
                    }, 12.5 * i);
                }
                chat_title.innerText = 'Виртуальный помощник';
                chat_close_btn.innerText = 'X';
                chat_header.style.cursor = 'default';
                chat_title.style.cursor = 'default';
                is_chat_open = true;
            }
            if (is_close_btn_clicked) {
                is_close_btn_clicked = false;
                is_chat_open = false;
                for (let i = 0; i < 20; i++) {
                    setTimeout(function () {
                        chat_window.style.height = FULL_WINDOW - (FULL_WINDOW - CLOSE_WINDOW) / 20 * (i + 1) + 'px';
                    }, 12.5 * i);
                }
                chat_title.innerText = 'Нужна помощь?';
                chat_close_btn.innerText = '';
                chat_header.style.cursor = 'pointer';
                chat_title.style.cursor = 'pointer';
            }
        }, false);
        chat_close_btn.addEventListener('click', function() {
            is_close_btn_clicked = true;
        }, false);
        function addUserMessage(message) {
            chat_history = document.querySelector('.chat-history');
            message_row = document.createElement('div');
            message_row.className = 'chat-client-message';
            message_block = document.createElement('div');
            message_block.innerText = message;
            avatar = document.createElement('img');
            avatar.className = 'avatar';
            avatar.setAttribute('src', '/test/modules/support_chat/human.png');
            avatar.setAttribute('alt', 'avatar');
            message_row.appendChild(message_block);
            message_row.appendChild(avatar);
            chat_history.appendChild(message_row);
            chat_history.scrollTo(0, chat_history.scrollHeight);
        }
        function addBotMessage(message) {
            chat_history = document.querySelector('.chat-history');
            message_row = document.createElement('div');
            message_row.className = 'chat-support-message';
            message_block = document.createElement('div');
            message_block.innerText = message;
            avatar = document.createElement('img');
            avatar.className = 'avatar';
            avatar.setAttribute('src', '/test/modules/support_chat/avatar.png');
            avatar.setAttribute('alt', 'avatar');
            message_row.appendChild(avatar);
            message_row.appendChild(message_block);
            chat_history.appendChild(message_row);
            chat_history.scrollTo(0, chat_history.scrollHeight);
        }
        function sendMessage() {
            if (chat_input.value != '') {
                addUserMessage(chat_input.value);
                xhr = new XMLHttpRequest();
                xhr.open('POST', 'http://localhost/bot')
                xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
                xhr.onreadystatechange = function() {
                    if (xhr.readyState === 4 && xhr.status === 200) {
                        addBotMessage(xhr.responseText);
                    }
                }
                xhr.send('message=' + chat_input.value);
                chat_input.value = '';
            }
        }
        chat_send_btn.addEventListener('click', function () {
            sendMessage();
        }, false);
        chat_input.addEventListener('keydown', function (event) {
            if (event.keyCode == 13) {
                event.preventDefault();
                sendMessage();
            }
        })
    </script>
</body>
</html>