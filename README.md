# hackathon

Создание интеллектуального помощника, способного оперативно генерировать ответы из базы знаний и Q&A МФЦ Спб

## Инструкция по запуску
1. [Скачать и установить Python](https://www.python.org/downloads/)
2. Клонировать данный репозиторий:
```
git clone https://github.com/DogeOk/hackathon.git
```
3. Переместиться в каталог репозитория:
```
cd hackathon
```
4. (Рекомендуется) Создать виртуальное окружение:
```
python -m venv hackathon
```
5. Активировать виртуальное окружение (если было создано):
   + Windows:
   ```
   .\hackathon\Scripts\activate.ps1
   ```
   + macOS или Linux:
   ```
   source hackathon/bin/activate
   ```
6. Установить необходимые пакеты из файла `requirements.txt`:
```
pip install -r requirements.txt
```
7. Установить модель
```
python -m spacy download ru_core_news_sm
```
8. Запустить проект:
```
python manage.py runserver
```