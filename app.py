import pandas as pd
from datetime import datetime
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import PieChart, BarChart, Reference
from openpyxl.styles.differential import DifferentialStyle
from openpyxl.formatting.rule import Rule
import os
import logging
import subprocess
from pathlib import Path
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz

# Скачать необходимые библиотеки, если они отсутствуют
try:
    stopwords.words('russian')
except LookupError:
    nltk.download('stopwords')

# Настройка логирования
logging.basicConfig(filename='processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

### Вспомогательные функции ####

def extract_phone_numbers(text):
    """
    Извлечение и нормализация телефонных номеров из текста.
    """
    if pd.isna(text):
        return []
    
    text_str = str(text)

    patterns = [
        r'\+7\s?$?\d{3}$?\s?\d{3}[\s-]?\d{2}[\s-]?\d{2}',
        r'8\s?$?\d{3}$?\s?\d{3}[\s-]?\d{2}[\s-]?\d{2}',
        r'($?\d{3}$?)?\s?\d{3}[\s-]?\d{2}[\s-]?\d{2}'
    ]

    phones = []
    for pattern in patterns:
        matches = re.findall(pattern, text_str)
        for match in matches:
            clean_phone = re.sub(r'[\s()-]', '', match)
            
            # Приводим номер телефона к стандартному виду (+7XXXXXXXXXX)
            if clean_phone.startswith('8'):
                clean_phone = '+7' + clean_phone[1:]
            elif clean_phone.startswith('7'):
                clean_phone = '+' + clean_phone
            
            if len(clean_phone) == 12 and clean_phone.startswith('+7'):
                phones.append(clean_phone)
        
    return list(set(phones))

def similar_text(text1, text2, threshold=0.85):
    """
    Проверка схожести двух строк с использованием коэффициента сходства.
    """
    if pd.isna(text1) or pd.isna(text2):
        return False
    return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio() >= threshold

def similar_phones(phones1, phones2):
    """
    Определение совпадения телефонных номеров.
    """
    if not phones1 or not phones2:
        return False
    return bool(set(phones1) & set(phones2))

def extract_address(text):
    """
    Выделение адреса из текста обращения.
    """
    if pd.isna(text):
        return 'Адрес не указан'

    text_str = str(text).lower()

    patterns = [
        r'(ул\.?|улица|просп|пр\.|проспект|пер\.|переулок|шоссе|наб\.|набережная)\s+\S+',
        r'д\.?\s*\d+',      # дом
        r'кв\.?\s*\d+'      # квартира
    ]

    parts = []
    for pattern in patterns:
        matches = re.findall(pattern, text_str)
        for match in matches:
            clean_match = re.sub(r'\s+', ' ', match).strip()
            parts.append(clean_match)

    return ' '.join(parts) if parts else 'Адрес не указан'

def categorize_consumer(consumer_name, appeal_text):
    """
    Категория потребителя ("физлицо"/"юрлицо").
    """
    if pd.isna(consumer_name):
        consumer_name = ''
    if pd.isna(appeal_text):
        appeal_text = ''

    consumer_str = str(consumer_name).lower()
    appeal_str = str(appeal_text).lower()

    keywords = [
        'ооо', 'зао', 'оао', 'ао', 'пао', 'нао', 'мку', 'мкуп', 'муп', 'гбу',
        'мбу', 'мбоу', 'мадоу', 'гкоу', 'учреждение', 'предприятие', 'компания',
        'фирма', 'корпус', 'строй', 'ремонт', 'сервис', 'центр', 'агентство',
        'холдинг', 'групп', 'торг', 'пром', 'завод', 'фабрика', 'комбинат',
        'управление', 'администрация', 'министерство', 'департамент'
    ]

    for kw in keywords:
        if kw in consumer_str or kw in appeal_str:
            return 'Юридическое лицо'

    return 'Физическое лицо'

### Основные функции работы с данными ####

def get_file_path():
    """
    Получение файла Excel через браузер с помощью Streamlit.
    """
    uploaded_file = st.file_uploader("Загрузите файл Excel с обращениями:")
    if uploaded_file is not None:
        file_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    else:
        return None

def load_data(file_path):
    """
    Загрузка данных из Excel и обработка потенциальных ошибок.
    """
    try:
        st.write("Этап 1: Загрузка данных...")
        df = pd.read_excel(file_path)
        st.success(f"Данные успешно загружены! Размер данных: {df.shape}.")
        logging.info(f"Успешная загрузка файла: {file_path}, размер: {df.shape}")

        # Попытка автоматического распознавания столбца с датой
        possible_date_cols = [col for col in df.columns if 'дата' in col.lower()]
        if possible_date_cols:
            date_col = possible_date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col])
            st.write(f"Считана дата из столбца: {date_col}")
        else:
            st.warning("Автоматически определить столбец с датой не удалось.")

        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        st.error(f"Ошибка при загрузке файла: {e}")
        return None

def download_file(file_path):
    """
    Предоставляет пользователю ссылку для скачивания сформированного отчёта.
    """
    try:
        st.write(f"Отчёт доступен для скачивания:", unsafe_allow_html=True)
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{Path(file_path).read_bytes().decode("latin1")}" download="{Path(file_path).name}">Нажмите сюда, чтобы скачать отчёт.</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Ошибка при подготовке отчёта для скачивания: {e}")

def generate_report_final(original_df, unique_df, duplicates_detailed, work_df, column_mapping):
    """
    Генерация финального отчёта в формате Excel.
    """
    try:
        st.write("Этап 4: Создание итогового отчёта...")

        output_path = Path('отчет_обращений_финал.xlsx')

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Далее идёт ваша оригинальная логика построения отчёта (создание листов, графики и т.п.)
            pass  # Оставляйте тут свою реализацию генерации отчёта!

        st.success(f"Отчёт успешно сформирован и доступен для скачивания!")
        download_file(output_path)

        return True
    except Exception as e:
        logging.error(f"Ошибка при создании отчёта: {e}")
        st.error(f"Ошибка при создании отчёта: {e}")
        return False

### Главный сценарий работы приложения ###

import streamlit as st

st.title("Анализ обращений клиентов")

# Выбор файла
uploaded_file = get_file_path()

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        # Остальная логика обработки данных (например, очистка, категоризация потребителей, выявление дублей и т.д.) должна быть размещена здесь.
        # После завершения основного процесса создаем отчёт:
        generate_report_final(df, ...)  # Передайте аргументы вашей функции согласно её реализации
