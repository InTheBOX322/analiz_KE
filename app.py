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
import re  # Добавлено: отсутствовал импорт re
from difflib import SequenceMatcher  # Добавлено: отсутствовал импорт
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz

# Download stopwords if not already downloaded
try:
    stopwords.words('russian')
except LookupError:
    nltk.download('stopwords')

# Настройка логирования
logging.basicConfig(
    filename='processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_phone_numbers(text):
    """Извлечение и нормализация телефонных номеров"""
    if pd.isna(text):
        return []

    text_str = str(text)

    # Паттерны для поиска телефонных номеров
    phone_patterns = [
        r'\+7\s?\(?\d{3}\)?\s?\d{3}[\s-]?\d{2}[\s-]?\d{2}',  # +7 форматы
        r'8\s?\(?\d{3}\)?\s?\d{3}[\s-]?\d{2}[\s-]?\d{2}',   # 8 форматы
        r'\(\d{3}\)\s?\d{3}[\s-]?\d{2}[\s-]?\d{2}',         # (999) 999-99-99
        r'\d{3}[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}',          # 999-99-99-99
        r'\d{3}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}',          # 999-999-99-99
    ]

    phones = []
    for pattern in phone_patterns:
        matches = re.findall(pattern, text_str)
        for match in matches:
            # Нормализация номера к формату +7XXXXXXXXXX
            clean_phone = re.sub(r'[\s\(\)\-+]', '', match)
            if clean_phone.startswith('8') and len(clean_phone) == 11:
                clean_phone = '+7' + clean_phone[1:]
            elif clean_phone.startswith('7') and len(clean_phone) == 11:
                clean_phone = '+7' + clean_phone[1:]
            elif len(clean_phone) == 10:
                clean_phone = '+7' + clean_phone

            if len(clean_phone) == 12 and clean_phone.startswith('+7'):
                phones.append(clean_phone)

    return list(set(phones))  # Убираем дубликаты

def similar_text(text1, text2, threshold=0.85):
    """Сравнение текстов с учетом возможных опечаток"""
    if pd.isna(text1) or pd.isna(text2):
        return False
    return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio() >= threshold

def similar_phones(phones1, phones2):
    """Сравнение списков телефонных номеров"""
    if not phones1 or not phones2:
        return False

    # Сравниваем номера телефонов
    common_phones = set(phones1) & set(phones2)
    return len(common_phones) > 0

def extract_address(text):
    """Извлечение адреса из текста обращения"""
    if pd.isna(text):
        return 'адрес не найден'

    text_str = str(text).lower()

    # Паттерны для поиска адреса
    address_patterns = [
        r'(ул\.?|улица|просп|пр\.|проспект|пер\.|переулок|шоссе|наб\.|набережная)[\s\w\d.-]+',
        r'д\.?\s*\d+',
        r'дом\.?\s*\d+',
        r'кв\.?\s*\d+',
        r'квартира\.?\s*\d+',
        r'корп\.?\s*\d+',
        r'строен\.?\s*\d+',
        r'микрорайон\.?\s*[\w\d]+',
        r'р-н\.?\s*[\w\d]+'
    ]

    address_parts = []
    for pattern in address_patterns:
        matches = re.findall(pattern, text_str, re.IGNORECASE)
        for match in matches:
            clean_match = re.sub(r'\s+', ' ', match).strip()
            address_parts.append(clean_match)

    unique_parts = []
    for part in address_parts:
        if part not in unique_parts:
            unique_parts.append(part)

    return ' '.join(unique_parts) if unique_parts else 'адрес не найден'

def categorize_consumer(consumer_name, appeal_text):
    """Определение категории потребителя"""
    if pd.isna(consumer_name):
        consumer_name = ''
    if pd.isna(appeal_text):
        appeal_text = ''

    consumer_str = str(consumer_name).lower()
    appeal_str = str(appeal_text).lower()

    legal_keywords = [
        'ооо', 'зао', 'оао', 'ао', 'пао', 'нао', 'мку', 'мкуп', 'муп', 'гбу',
        'мбу', 'мбоу', 'мадоу', 'гкоу', 'учреждение', 'предприятие', 'компания',
        'фирма', 'корпус', 'строй', 'ремонт', 'сервис', 'центр', 'агентство',
        'холдинг', 'групп', 'торг', 'пром', 'завод', 'фабрика', 'комбинат',
        'управление', 'администрация', 'министерство', 'департамент'
    ]

    for keyword in legal_keywords:
        if keyword in consumer_str:
            return 'Юридическое лицо'

    for keyword in legal_keywords:
        if keyword in appeal_str:
            return 'Юридическое лицо'

    if consumer_str.strip() and appeal_str.strip():
        return 'Физическое лицо'

    return 'Не определено'

def get_file_path():
    """Получение пути к файлу через стандартный ввод"""
    try:
        file_path = input("Пожалуйста, введите путь к файлу Excel с обращениями: ").strip()
        
        if not file_path:
            print("Файл не выбран. Программа завершена.")
            return None
        
        if not os.path.exists(file_path):
            print(f"Файл '{file_path}' не найден.")
            return None
            
        print(f"Файл '{file_path}' успешно загружен.")
        return file_path
        
    except Exception as e:
        logging.error(f"Ошибка при загрузке файла: {e}")
        print(f"\nОшибка при загрузке файла: {e}")
        return None

def load_data(file_path):
    """Загрузка данных с обработкой ошибок"""
    try:
        print("\nЭтап 1: Загрузка данных...")
        df = pd.read_excel(file_path)
        print(f"Данные успешно загружены! Размер данных: {df.shape}")
        print(f"Столбцы в файле: {list(df.columns)}")
        logging.info(f"Успешная загрузка файла: {file_path}")
        logging.info(f"Столбцы в файле: {list(df.columns)}")

        # Автоматическое определение столбца с датой
        date_columns = []
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in ['дата', 'date', 'создан', 'created']):
                date_columns.append(col)

        if date_columns:
            print(f"Найдены потенциальные столбцы с датой: {date_columns}")
            # Берем первый подходящий столбец
            date_column = date_columns[0]
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            print(f"Используется столбец с датой: '{date_column}'")
        else:
            print("⚠️ Столбец с датой не найден автоматически")

        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        print(f"\nОшибка при загрузке файла: {e}")
        return None

# ... (остальные функции остаются без изменений, кроме download_file)

def download_file(file_path):
    """Функция для загрузки файла (альтернатива files.download)"""
    try:
        # Для локального использования просто выводим путь к файлу
        print(f"Файл сохранен по пути: {os.path.abspath(file_path)}")
        
        # Если запущено в среде, поддерживающей загрузку файлов
        try:
            from google.colab import files as colab_files
            colab_files.download(file_path)
            print(f"Файл '{file_path}' загружен через Colab.")
        except ImportError:
            print("Для автоматической загрузки файла запустите код в Google Colab")
            
    except Exception as e:
        print(f"Ошибка при загрузке файла: {e}")

# В функции generate_report_final замените вызов files.download на download_file
def generate_report_final(original_df, unique_df, duplicates_detailed, work_df, column_mapping):
    """Creation of the final report with enhanced statistics and detailed duplicate analysis."""
    try:
        print("\nЭтап 4: Формирование итогового отчета...")

        output_path = Path('отчет_обращений_финал.xlsx')

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # ... (остальной код функции без изменений)

        print(f"\n📊 Отчет успешно сохранен по пути: {output_path.absolute()}")
        print(f"📋 Листы отчета:")
        print(f"   1. Исходные данные - {len(original_df)} записей")
        print(f"   2. Уникальные - {len(unique_df)} записей")
        print(f"   3. Дубликаты - {len(duplicates_detailed)} записей")
        print(f"   4. Статистика - аналитическая информация")
        print(f"   5. Графики анализа - визуализация данных")
        if len(duplicates_detailed) > 0:
             print(f"   6. Детальный анализ дубликатов - подробная информация по группам")

        logging.info(f"Успешное создание отчета: {output_path}")

        # Замените files.download на download_file
        download_file(output_path)
        print(f"\n⬇️ Файл '{output_path.name}' готов к загрузке.")

        return True

    except Exception as e:
        logging.error(f"Ошибка при создании отчета: {e}")
        print(f"\n❌ Ошибка при формировании отчета: {e}")
        return False

# ... (остальной код без изменений)

# Уберите вызов main_final() в конце, если хотите контролировать запуск
if __name__ == "__main__":
    main_final()
