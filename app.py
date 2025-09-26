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
from google.colab import files # Import files module
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

# Keeping SequenceMatcher for address comparison in format_excel_report
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
    """Получение пути к файлу через загрузку пользователя"""
    try:
        print("Пожалуйста, загрузите файл Excel с обращениями:")
        uploaded = files.upload()

        if not uploaded:
            print("Файл не выбран. Программа завершена.")
            return None

        # Assuming only one file is uploaded
        file_name = list(uploaded.keys())[0]
        print(f"Файл '{file_name}' успешно загружен.")

        return file_name
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

def validate_data(df):
    """Проверка корректности данных"""
    try:
        print("\nЭтап 2: Проверка данных...")

        # Выводим все столбцы для информации
        print("Столбцы в данных:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i}. {col}")

        # Проверяем наличие ключевых столбцов
        required_keywords = {
            'потребитель': ['потребитель', 'клиент', 'заявитель'],
            'текст обращения': ['обращен', 'текст', 'описан', 'жалоб'],
            'адрес': ['адрес', 'местополож', 'объект']
        }

        found_columns = {}
        for key, keywords in required_keywords.items():
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in keywords):
                    found_columns[key] = col
                    break

        print("\nНайденные ключевые столбцы:")
        for key, col in found_columns.items():
            print(f"  {key}: '{col}'")

        # Проверка на пустые значения
        key_columns = list(found_columns.values())
        if key_columns:
            empty_count = df[key_columns].isnull().sum()
            if empty_count.any():
                print(f"Предупреждение: Найдены пустые значения:\n{empty_count}")

        print("Проверка данных завершена успешно!")
        logging.info("Успешная валидация данных")
        return df, found_columns

    except Exception as e:
        logging.error(f"Ошибка валидации данных: {e}")
        print(f"\nОшибка валидации: {e}")
        return None, None

def find_date_column(df):
    """Поиск столбца с датой"""
    for col in df.columns:
        if any(keyword in str(col).lower() for keyword in ['дата', 'date', 'создан', 'created']):
            return col
    return None

# --- Advanced Text Similarity (TF-IDF) ---
def tfidf_similarity(text1, text2, stop_words=None):
    """
    Calculate TF-IDF cosine similarity between two texts.
    """
    if pd.isna(text1) or pd.isna(text2):
        return 0.0

    text1_str = str(text1).lower()
    text2_str = str(text2).lower()

    corpus = [text1_str, text2_str]

    # Use Russian stop words
    if stop_words is None:
        stop_words = stopwords.words('russian')

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError: # Handle empty vocabulary case
        return 0.0


    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # The similarity between text1 and text2 is at index [0, 1]
    return similarity_matrix[0, 1]

# --- Fuzzy Matching for Names and Addresses ---
def fuzzy_similar_names(name1, name2, threshold=80):
    """
    Compares two names using fuzzy matching (token_set_ratio).
    Returns True if similarity is above the threshold, False otherwise.
    """
    if pd.isna(name1) or pd.isna(name2):
        return False
    score = fuzz.token_set_ratio(str(name1).lower(), str(name2).lower())
    return score >= threshold

def fuzzy_similar_addresses(address1, address2, threshold=75):
    """
    Compares two addresses using fuzzy matching (token_sort_ratio).
    Returns True if similarity is above the threshold, False otherwise.
    """
    if pd.isna(address1) or pd.isna(address2):
        return False
    score = fuzz.token_sort_ratio(str(address1).lower(), str(address2).lower())
    return score >= threshold


# --- Weighted Duplicate Detection Logic ---
def find_unique_and_duplicates_weighted(df, column_mapping, tfidf_threshold=0.4, name_fuzzy_threshold=80, address_fuzzy_threshold=75, res_fuzzy_threshold=85, filial_fuzzy_threshold=85, score_threshold=15, criteria_weights=None):
    """Enhanced algorithm using weighted scoring for duplicate detection."""
    try:
        print("\nЭтап 3: Определение уникальности (взвешенная система)...")

        # Define default weights if none provided
        if criteria_weights is None:
             criteria_weights = {
                 'телефоны': 10,
                 'текст (TF-IDF)': 8,
                 'потребитель (fuzzy)': 7,
                 'адрес (fuzzy)': 6,
                 'тип обращения (exact)': 5,
                 'тематика обращения (exact)': 5,
                 'дата': 4,
                 'РЭС (fuzzy/exact)': 3,
                 'Филиал (fuzzy/exact)': 2
             }


        # Получаем названия столбцов из mapping
        consumer_col = column_mapping.get('потребитель', 'Потребитель')
        appeal_col = column_mapping.get('текст обращения', 'Текст обращения')
        address_col = column_mapping.get('адрес', 'Адрес')

        # Identify additional columns from the DataFrame itself if not in mapping
        type_appeal_col = 'Тип обращения' if 'Тип обращения' in df.columns else None
        subject_appeal_col = 'Тематика обращения' if 'Тематика обращения' in df.columns else None
        res_col = 'РЭС' if 'РЭС' in df.columns else None
        filial_col = 'Филиал' if 'Филиал' in df.columns else None


        # Находим столбец с датой
        date_col = find_date_column(df)
        if not date_col:
            print("❌ Не найден столбец с датой! Проверка уникальности невозможна.")
            return df, pd.DataFrame(), df

        print(f"Используемые столбцы:")
        print(f"  Потребитель: '{consumer_col}' (порог fuzzy: {name_fuzzy_threshold})")
        print(f"  Текст обращения: '{appeal_col}' (порог TF-IDF: {tfidf_threshold})")
        print(f"  Адрес: '{address_col}' (порог fuzzy: {address_fuzzy_threshold})")
        print(f"  Дата: '{date_col}'")
        if type_appeal_col: print(f"  Тип обращения: '{type_appeal_col}' (точное совпадение)")
        if subject_appeal_col: print(f"  Тематика обращения: '{subject_appeal_col}' (точное совпадение)")
        if res_col: print(f"  РЭС: '{res_col}' (порог fuzzy: {res_fuzzy_threshold})")
        if filial_col: print(f"  Филиал: '{filial_col}' (порог fuzzy: {filial_fuzzy_threshold})")
        print(f"\nПорог суммарного балла для дубликата: {score_threshold}")
        print(f"Веса критериев: {criteria_weights}")


        # Создаем рабочую копию
        work_df = df.copy()

        # Извлекаем адреса и телефоны из текста обращений
        print("Извлечение адресов и телефонов из текста обращений...")
        work_df['Извлеченный адрес'] = work_df[appeal_col].apply(extract_address)
        work_df['Телефоны'] = work_df[appeal_col].apply(extract_phone_numbers)
        work_df['Количество телефонов'] = work_df['Телефоны'].apply(len)

        # Добавляем нормализованные данные для сравнения
        work_df['Текст_normalized'] = work_df[appeal_col].astype(str).str.lower().str.strip()
        work_df['Потребитель_normalized'] = work_df[consumer_col].astype(str).str.lower().str.strip()
        work_df['Адрес_normalized'] = work_df[address_col].astype(str).str.lower().str.strip()

        # Handle NaT values when extracting date components
        work_df['Дата_день'] = work_df[date_col].dt.date.apply(lambda x: x if pd.notna(x) else None)
        work_df['Дата_неделя'] = work_df[date_col].dt.to_period('W').apply(lambda r: r.start_time.date() if pd.notna(r) else None)
        work_df['Дата_месяц'] = work_df[date_col].dt.to_period('M').apply(lambda r: r.start_time.date() if pd.notna(r) else None)

        # Определяем категорию потребителя (already present, keeping it)
        work_df['Категория потребителя'] = work_df.apply(
            lambda row: categorize_consumer(row[consumer_col], row[appeal_col]), axis=1
        )

        # Создаем колонку для группировки - основной ключ для первичной группировки (using normalized consumer name)
        # Ensure date components are not None before concatenating
        work_df['group_key'] = work_df['Потребитель_normalized'] + '|' + work_df['Дата_неделя'].astype(str).replace('None', '')


        # Определяем уникальные записи
        work_df['Тип записи'] = 'Уникальная'
        work_df['Группа дубликатов'] = ''
        work_df['Критерий совпадения'] = ''
        work_df['Суммарный балл совпадения'] = 0.0 # New column for total score

        print("Поиск дубликатов с учетом взвешенных критериев...")

        # Группируем по потребителю и неделе
        group_keys = work_df['group_key'].unique()
        total_groups = len(group_keys)

        duplicate_group_counter = 0

        for i, group_key in enumerate(group_keys):
            group_indices_mask = (work_df['group_key'] == group_key)
            group_indices = work_df.index[group_indices_mask].tolist()


            if len(group_indices) > 1:
                if (i + 1) % 100 == 0:
                    print(f"Обработано групп: {i + 1}/{total_groups}")

                # Iterate through the group to find duplicates within it
                for i_idx in range(len(group_indices)):
                    idx_i = group_indices[i_idx]
                    # If this record is already marked as a duplicate (and not the primary in a new group), skip it
                    if work_df.loc[idx_i, 'Тип записи'] == 'Дубликат':
                        continue

                    # Get values using .loc
                    consumer_i = work_df.loc[idx_i, consumer_col]
                    text_i = work_df.loc[idx_i, appeal_col]
                    address_i = work_df.loc[idx_i, address_col]
                    phones_i = work_df.loc[idx_i, 'Телефоны']
                    date_i = work_df.loc[idx_i, 'Дата_день']
                    extracted_address_i = work_df.loc[idx_i, 'Извлеченный адрес']

                    # Get values for additional columns
                    type_appeal_i = work_df.loc[idx_i, type_appeal_col] if type_appeal_col else None
                    subject_appeal_i = work_df.loc[idx_i, subject_appeal_col] if subject_appeal_col else None
                    res_i = work_df.loc[idx_i, res_col] if res_col else None
                    filial_i = work_df.loc[idx_i, filial_col] if filial_col else None


                    for j_idx in range(i_idx + 1, len(group_indices)):
                        idx_j = group_indices[j_idx]
                        # If this record is already marked as a duplicate, skip it
                        if work_df.loc[idx_j, 'Тип записи'] == 'Дубликат':
                            continue

                        # Get values using .loc
                        consumer_j = work_df.loc[idx_j, consumer_col]
                        text_j = work_df.loc[idx_j, appeal_col]
                        address_j = work_df.loc[idx_j, address_col]
                        phones_j = work_df.loc[idx_j, 'Телефоны']
                        date_j = work_df.loc[idx_j, 'Дата_день']
                        extracted_address_j = work_df.loc[idx_j, 'Извлеченный адрес']

                        # Get values for additional columns
                        type_appeal_j = work_df.loc[idx_j, type_appeal_col] if type_appeal_col else None
                        subject_appeal_j = work_df.loc[idx_j, subject_appeal_col] if subject_appeal_col else None
                        res_j = work_df.loc[idx_j, res_col] if res_col else None
                        filial_j = work_df.loc[idx_j, filial_col] if filial_col else None


                        # Проверяем различные критерии совпадения и накапливаем балл
                        match_criteria = []
                        total_score = 0.0

                        # 1. Совпадение телефонов
                        if similar_phones(phones_i, phones_j):
                            match_criteria.append('телефоны')
                            total_score += criteria_weights.get('телефоны', 0)

                        # 2. Совпадение текста (Используем TF-IDF)
                        text_tfidf_score = tfidf_similarity(text_i, text_j)
                        if text_tfidf_score > tfidf_threshold:
                            match_criteria.append(f'текст (TF-IDF)')
                            # Optionally, scale weight by TF-IDF score, but using fixed weight for simplicity now
                            total_score += criteria_weights.get('текст (TF-IDF)', 0)

                        # 3. Совпадение адреса (Используем fuzzy matching on original address and extracted address)
                        if fuzzy_similar_addresses(address_i, address_j, threshold=address_fuzzy_threshold) or fuzzy_similar_addresses(extracted_address_i, extracted_address_j, threshold=address_fuzzy_threshold):
                             match_criteria.append(f'адрес (fuzzy)')
                             total_score += criteria_weights.get('адрес (fuzzy)', 0)

                        # 4. Совпадение даты (в пределах 3 дней)
                        if pd.notna(date_i) and pd.notna(date_j):
                            date_diff = abs((date_i - date_j).days)
                            if date_diff <= 3:
                                match_criteria.append('дата')
                                total_score += criteria_weights.get('дата', 0)

                        # 5. Совпадение потребителя (Используем fuzzy matching on original consumer name)
                        if fuzzy_similar_names(consumer_i, consumer_j, threshold=name_fuzzy_threshold):
                             match_criteria.append(f'потребитель (fuzzy)')
                             total_score += criteria_weights.get('потребитель (fuzzy)', 0)

                        # 6. Совпадение Типа обращения (точное совпадение)
                        if type_appeal_col and pd.notna(type_appeal_i) and pd.notna(type_appeal_j) and str(type_appeal_i).lower() == str(type_appeal_j).lower():
                             match_criteria.append(f'тип обращения (exact)')
                             total_score += criteria_weights.get('тип обращения (exact)', 0)

                        # 7. Совпадение Тематики обращения (точное совпадение)
                        if subject_appeal_col and pd.notna(subject_appeal_i) and pd.notna(subject_appeal_j) and str(subject_appeal_i).lower() == str(subject_appeal_j).lower():
                             match_criteria.append(f'тематика обращения (exact)')
                             total_score += criteria_weights.get('тематика обращения (exact)', 0)

                        # 8. Совпадение РЭС (fuzzy or exact match)
                        if res_col and (
                            (pd.notna(res_i) and pd.notna(res_j) and str(res_i).lower() == str(res_j).lower()) or
                            fuzzy_similar_addresses(res_i, res_j, threshold=res_fuzzy_threshold) # Using address fuzzy logic for RЭС names
                        ):
                             match_criteria.append(f'РЭС (fuzzy/exact)')
                             total_score += criteria_weights.get('РЭС (fuzzy/exact)', 0)


                        # 9. Совпадение Филиала (fuzzy or exact match)
                        if filial_col and (
                            (pd.notna(filial_i) and pd.notna(filial_j) and str(filial_i).lower() == str(filial_j).lower()) or
                            fuzzy_similar_addresses(filial_i, filial_j, threshold=filial_fuzzy_threshold) # Using address fuzzy logic for Filial names
                        ):
                             match_criteria.append(f'Филиал (fuzzy/exact)')
                             total_score += criteria_weights.get('Филиал (fuzzy/exact)', 0)


                        # Determine if it's a duplicate based on the total score threshold
                        is_duplicate = total_score >= score_threshold


                        if is_duplicate:
                            # If the primary record hasn't been assigned a group yet, assign one
                            if work_df.loc[idx_i, 'Тип записи'] == 'Уникальная':
                                duplicate_group_counter += 1
                                group_id = f"Группа_{duplicate_group_counter}"
                                work_df.loc[idx_i, 'Тип записи'] = 'Основной в группе'
                                work_df.loc[idx_i, 'Группа дубликатов'] = group_id
                                work_df.loc[idx_i, 'Критерий совпадения'] = ', '.join(match_criteria)
                                work_df.loc[idx_i, 'Суммарный балл совпадения'] = total_score # Assign score to primary

                            else:
                                # If the primary record already has a group, use that group_id
                                group_id = work_df.loc[idx_i, 'Группа дубликатов']
                                # Append criteria if not already present
                                existing_criteria = work_df.loc[idx_i, 'Критерий совпадения'].split(', ')
                                new_criteria = list(set(existing_criteria + match_criteria))
                                work_df.loc[idx_i, 'Критерий совпадения'] = ', '.join(new_criteria)
                                # Update primary score if current score is higher (optional, or take max/avg)
                                work_df.loc[idx_i, 'Суммарный балл совпадения'] = max(work_df.loc[idx_i, 'Суммарный балл совпадения'], total_score)


                            # Mark the second record as a duplicate and assign it to the same group
                            work_df.loc[idx_j, 'Тип записи'] = 'Дубликат'
                            work_df.loc[idx_j, 'Группа дубликатов'] = group_id
                            work_df.loc[idx_j, 'Критерий совпадения'] = ', '.join(match_criteria)
                            work_df.loc[idx_j, 'Суммарный балл совпадения'] = total_score # Assign score to duplicate

        # Create final DataFrames
        unique_df = work_df[work_df['Тип записи'].isin(['Уникальная', 'Основной в группе'])].copy()
        duplicates_detailed = work_df[work_df['Тип записи'] == 'Дубликат'].copy()

        # Remove technical columns from final tables
        columns_to_drop = ['Текст_normalized', 'Потребитель_normalized', 'Адрес_normalized', 'Дата_день',
                          'Дата_неделя', 'Дата_месяц', 'group_key', 'Извлеченный адрес', 'Телефоны']

        final_columns = [col for col in unique_df.columns if col not in columns_to_drop]
        unique_df = unique_df[final_columns]
        duplicates_detailed = duplicates_detailed[final_columns]

        # Sort duplicates by group ID and score for better visualization
        duplicates_detailed = duplicates_detailed.sort_values(by=['Группа дубликатов', 'Суммарный балл совпадения'], ascending=[True, False]).reset_index(drop=True)


        print(f"✅ Уникальные обращения: {len(unique_df)}")
        print(f"✅ Дубликатов (записей): {len(duplicates_detailed)}")
        print(f"✅ Из них основных записей: {len(unique_df[unique_df['Тип записи'] == 'Основной в группе'])}")
        num_duplicate_groups = work_df['Группа дубликатов'].nunique() - (work_df['Группа дубликатов'] == '').sum() # Count non-empty group IDs
        print(f"✅ Дубликатов (группы): {num_duplicate_groups}")

        logging.info(f"Уникальных обращений: {len(unique_df)}, Дубликатов: {len(duplicates_detailed)}")
        return unique_df, duplicates_detailed, work_df

    except Exception as e:
        logging.error(f"Ошибка при определении уникальности: {e}")
        print(f"\n❌ Ошибка при обработке данных: {e}")
        return df, pd.DataFrame(), df


def create_detailed_statistics_enhanced(original_df, unique_df, duplicates_detailed, work_df):
    """Creation of enhanced detailed statistics with new duplicate analysis."""
    try:
        print("\nСоздание детальной статистики (расширенная)...")

        stats_data = []

        # Basic Statistics
        total_appeals = len(original_df)
        total_unique = len(unique_df)
        total_duplicates = len(duplicates_detailed)
        num_duplicate_groups = work_df['Группа дубликатов'].nunique() - (work_df['Группа дубликатов'] == '').sum() # Count non-empty group IDs


        stats_data.append({'Категория': 'ОБЩАЯ СТАТИСТИКА', 'Показатель': 'Всего обращений', 'Значение': total_appeals})
        stats_data.append({'Категория': 'ОБЩАЯ СТАТИСТИКА', 'Показатель': 'Уникальных обращений', 'Значение': total_unique})
        stats_data.append({'Категория': 'ОБЩАЯ СТАТИСТИКА', 'Показатель': 'Дубликатов (записей)', 'Значение': total_duplicates})
        stats_data.append({'Категория': 'ОБЩАЯ СТАТИСТИКА', 'Показатель': 'Дубликатов (группы)', 'Значение': num_duplicate_groups})
        stats_data.append({'Категория': 'ОБЩАЯ СТАТИСТИКА', 'Показатель': 'Процент дубликатов (от записей)',
                          'Значение': f"{(total_duplicates/total_appeals*100):.1f}%" if total_appeals > 0 else '0%'})
        if num_duplicate_groups > 0:
             stats_data.append({'Категория': 'ОБЩАЯ СТАТИСТИКА', 'Показатель': 'Средний размер группы дубликатов',
                               'Значение': f"{(duplicates_detailed.groupby('Группа дубликатов').size().mean() + 1):.2f}" if num_duplicate_groups > 0 else 'N/A'}) # +1 to include the primary record


        # Statistics by Consumer Categories
        if 'Категория потребителя' in work_df.columns:
            category_stats = work_df['Категория потребителя'].value_counts()
            for category, count in category_stats.items():
                stats_data.append({
                    'Категория': 'КАТЕГОРИИ ПОТРЕБИТЕЛЕЙ',
                    'Показатель': category,
                    'Значение': count
                })

        # Statistics by Record Types
        if 'Тип записи' in work_df.columns:
            type_stats = work_df['Тип записи'].value_counts()
            for type_name, count in type_stats.items():
                stats_data.append({
                    'Категория': 'ТИПЫ ЗАПИСЕЙ',
                    'Показатель': type_name,
                    'Значение': count
                })

        # Statistics by Matching Criteria
        if 'Критерий совпадения' in duplicates_detailed.columns:
            # Analyze individual criteria frequency
            all_criteria = duplicates_detailed['Критерий совпадения'].str.split(', ').explode()
            # Filter out empty strings and 'nan' string which can appear from splitting
            all_criteria = all_criteria[all_criteria.str.strip() != ''].dropna()
            criteria_stats = all_criteria.value_counts().head(15) # Show top 15 criteria

            if not criteria_stats.empty:
                 stats_data.append({'Категория': 'ЧАСТОТА КРИТЕРИЕВ СОВПАДЕНИЯ ДУБЛИКАТОВ', 'Показатель': 'Всего совпадений критериев', 'Значение': len(all_criteria)})
                 for criteria, count in criteria_stats.items():
                     stats_data.append({
                         'Категория': 'ЧАСТОТА КРИТЕРИЕВ СОВПАДЕТАИЯ ДУБЛИКАТОВ',
                         'Показатель': criteria,
                         'Значение': count
                     })

            # Analyze common criteria combinations (optional, can be verbose)
            # Let's skip combinations for now to keep the statistics table manageable

        # Statistics by Date
        date_col = find_date_column(original_df)
        if date_col:
            # By Month
            # Ensure date column is datetime objects before formatting
            original_df[date_col] = pd.to_datetime(original_df[date_col], errors='coerce')
            # Filter out NaT values before calculating monthly stats
            monthly_stats = original_df[original_df[date_col].notna()][date_col].dt.to_period('M').value_counts().sort_index()
            for month, count in monthly_stats.items():
                stats_data.append({
                    'Категория': 'СТАТИСТИКА ПО МЕСЯЦАМ',
                    'Показатель': f'{month}',
                    'Значение': count
                })

            # By Day of Week
            # Filter out NaT values before calculating day stats
            original_df_valid_dates = original_df[original_df[date_col].notna()].copy()
            original_df_valid_dates['День недели'] = original_df_valid_dates[date_col].dt.day_name()
            day_stats = original_df_valid_dates['День недели'].value_counts()
            # Order days of the week (optional, depends on desired order)
            # days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            # day_stats = day_stats.reindex(days_order).fillna(0) # Uncomment to enforce order
            for day, count in day_stats.items():
                stats_data.append({
                    'Категория': 'СТАТИСТИКА ПО ДНЯМ НЕДЕЛИ',
                    'Показатель': day,
                    'Значение': count
                })

        # Statistics by Number of Phones
        if 'Количество телефонов' in work_df.columns:
            phone_stats = work_df['Количество телефонов'].value_counts().sort_index()
            for count_phones, records in phone_stats.items():
                stats_data.append({
                    'Категория': 'СТАТИСТИКА ПО ТЕЛЕФОНАМ',
                    'Показатель': f'Обращений с {count_phones} телефоном(ами)',
                    'Значение': records
                })

        # Statistics by Duplicate Group Size
        if num_duplicate_groups > 0:
             # Get the size of each duplicate group from work_df
             group_sizes = work_df[work_df['Группа дубликатов'] != '']['Группа дубликатов'].value_counts() + 1 # +1 because value_counts counts duplicates, add primary
             group_size_distribution = group_sizes.value_counts().sort_index()

             if not group_size_distribution.empty:
                  stats_data.append({'Категория': 'РАСПРЕДЕЛЕНИЕ РАЗМЕРА ГРУПП ДУБЛИКАТОВ', 'Показатель': 'Всего групп', 'Значение': num_duplicate_groups})
                  for size, count in group_size_distribution.items():
                       stats_data.append({
                           'Категория': 'РАСПРЕДЕЛЕНИЕ РАЗМЕРА ГРУПП ДУБЛИКАТОВ',
                           'Показатель': f'{size} запис(ь/и)',
                           'Значение': count
                       })


        stats_df = pd.DataFrame(stats_data)
        return stats_df

    except Exception as e:
        logging.error(f"Ошибка при создании статистики: {e}")
        print(f"❌ Ошибка при создании статистики: {e}")
        return pd.DataFrame()

def create_analysis_charts_enhanced(work_df, duplicates_detailed, writer):
    """Creation of enhanced analysis charts."""
    try:
        workbook = writer.book
        # Check if the sheet already exists and remove it if it does
        if 'Графики анализа' in workbook.sheetnames:
            workbook.remove(workbook['Графики анализа'])
        worksheet = workbook.create_sheet('Графики анализа')

        # Header
        worksheet['A1'] = 'АНАЛИТИЧЕСКИЕ ГРАФИКИ'
        worksheet['A1'].font = Font(bold=True, size=16)

        current_row = 3

        # Helper function to add data and chart
        def add_chart(data, title, chart_type, x_title, y_title, chart_cell="D", data_cols=2, data_start_col=1):
            nonlocal current_row
            if data.empty:
                # Add a message indicating no data for this chart
                worksheet.cell(row=current_row, column=1, value=f"Нет данных для графика: {title}")
                worksheet.cell(row=current_row, column=1).font = Font(italic=True, color="808080")
                current_row += 2
                return

            worksheet.cell(row=current_row, column=data_start_col, value=title)
            worksheet.cell(row=current_row, column=data_start_col).font = Font(bold=True)
            current_row += 1

            # Write data to sheet
            start_row = current_row
            if isinstance(data, pd.Series):
                 # For value_counts series
                 for i, (label, count) in enumerate(data.items()):
                     worksheet.cell(row=current_row, column=data_start_col, value=label)
                     worksheet.cell(row=current_row, column=data_start_col + 1, value=count)
                     current_row += 1
                 end_row = current_row - 1
                 labels_ref = Reference(worksheet, min_col=data_start_col, min_row=start_row, max_row=end_row)
                 data_ref = Reference(worksheet, min_col=data_start_col + 1, min_row=start_row, max_row=end_row)
            elif isinstance(data, pd.DataFrame) and data_cols > 0:
                 # For custom dataframes, assuming first column is labels, rest are values
                 # Write header if not already written by previous chart
                 # (Assuming simple data structure: label | value1 | value2 ...)
                 for r_idx, row in data.iterrows():
                      for c_idx in range(data_cols):
                           worksheet.cell(row=current_row, column=data_start_col + c_idx, value=row.iloc[c_idx])
                      current_row += 1
                 end_row = current_row - 1
                 labels_ref = Reference(worksheet, min_col=data_start_col, min_row=start_row, max_row=end_row)
                 data_ref = Reference(worksheet, min_col=data_start_col + 1, min_row=start_row, max_row=end_row, max_col=data_start_col + data_cols -1)


            # Create chart
            if chart_type == 'pie':
                chart = PieChart()
                chart.add_data(data_ref, titles_from_data=False)
                chart.set_categories(labels_ref)
                chart.title = title
            elif chart_type == 'bar':
                chart = BarChart()
                chart.add_data(data_ref, titles_from_data=False)
                chart.set_categories(labels_ref) # Corrected typo here, was categories_ref
                chart.title = title
                chart.x_axis.title = x_title
                chart.y_axis.title = y_title
                # Ensure x-axis labels are visible for many categories
                if len(data) > 10: # Heuristic for when labels might overlap
                     chart.x_axis.textRotation = 45


            # Add chart to worksheet
            worksheet.add_chart(chart, f"{chart_cell}{start_row}")
            current_row += max(len(data) + 2, 15) # Move to the next available row for the next chart

        # 1. Chart: Distribution by Consumer Categories
        if 'Категория потребителя' in work_df.columns:
            category_stats = work_df['Категория потребителя'].value_counts()
            add_chart(category_stats, 'Распределение по категориям потребителей', 'pie', '', '')

        # 2. Chart: Distribution by Record Types
        if 'Тип записи' in work_df.columns:
            type_stats = work_df['Тип записи'].value_counts()
            add_chart(type_stats, 'Распределение по типам записей', 'pie', '', '')

        # 3. Chart: Distribution by Month
        date_col = find_date_column(work_df)
        if date_col:
            # Ensure date column is datetime objects before formatting
            work_df[date_col] = pd.to_datetime(work_df[date_col], errors='coerce')
            # Filter out NaT values before calculating monthly stats
            monthly_stats = work_df[work_df[date_col].notna()][date_col].dt.to_period('M').value_counts().sort_index()
            # Convert Period to string for charting
            monthly_stats.index = monthly_stats.index.astype(str)
            add_chart(monthly_stats, 'Распределение по месяцам', 'bar', 'Месяц', 'Количество обращений')

        # 4. Chart: Distribution by Day of Week
        if date_col:
            # Filter out NaT values before calculating day stats
            work_df_valid_dates = work_df[work_df[date_col].notna()].copy()
            work_df_valid_dates['День недели'] = work_df_valid_dates[date_col].dt.day_name()
            day_stats = work_df_valid_dates['День недели'].value_counts()
            # Order days of the week
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_stats = day_stats.reindex(days_order).fillna(0)
            add_chart(day_stats, 'Распределение по дням недели', 'bar', 'День недели', 'Количество обращений')

        # 5. Chart: Distribution by Number of Phones in Appeal
        if 'Количество телефонов' in work_df.columns:
             phone_count_stats = work_df['Количество телефонов'].value_counts().sort_index()
             add_chart(phone_count_stats, 'Распределение по количеству телефонов в обращении', 'bar', 'Количество телефонов', 'Количество обращений')

        # --- New Charts for Duplicate Analysis ---

        # 6. Chart: Most Frequent Matching Criteria (Individual)
        if 'Критерий совпадения' in duplicates_detailed.columns and not duplicates_detailed.empty:
             all_criteria = duplicates_detailed['Критерий совпадения'].str.split(', ').explode()
             all_criteria = all_criteria[all_criteria.str.strip() != ''].dropna()
             criteria_counts = all_criteria.value_counts().head(10) # Show top 10 for the chart
             add_chart(criteria_counts, 'Топ-10 критериев совпадения в дубликатах', 'bar', 'Критерий', 'Частота')

        # 7. Chart: Distribution of Duplicate Group Sizes
        num_duplicate_groups = work_df['Группа дубликатов'].nunique() - (work_df['Группа дубликатов'] == '').sum()
        if num_duplicate_groups > 0:
            group_sizes = work_df[work_df['Группа дубликатов'] != '']['Группа дубликатов'].value_counts() + 1 # +1 for primary
            group_size_distribution = group_sizes.value_counts().sort_index()
            # Convert index to string for plotting if necessary
            group_size_distribution.index = group_size_distribution.index.astype(str)
            add_chart(group_size_distribution, 'Распределение групп по размеру', 'bar', 'Количество записей в группе', 'Количество групп')


        return True

    except Exception as e:
        logging.error(f"Ошибка при создании графиков: {e}")
        print(f"⚠️ Не удалось создать графики: {e}")
        return False


def format_excel_report(writer):
    """Форматирование Excel отчета"""
    try:
        workbook = writer.book

        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        center_align = Alignment(horizontal="center", vertical="center")
        wrap_text_align = Alignment(horizontal="center", vertical="center", wrap_text=True)

        # Define alternating fill colors
        fill1 = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid") # Light blue
        fill2 = PatternFill(start_color="B8CCE4", end_color="B8CCE4", fill_type="solid") # Slightly darker blue


        for sheet_name in workbook.sheetnames:
            worksheet = workbook[sheet_name]

            # Форматирование заголовков
            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = wrap_text_align # Use wrap_text_align for headers

            # Apply alternating row colors for duplicates
            if sheet_name == 'Дубликаты':
                group_column_index = None
                for col_index, cell in enumerate(worksheet[1]):
                    if cell.value == 'Группа дубликатов':
                        group_column_index = col_index + 1
                        break

                if group_column_index is not None:
                    current_group = None
                    current_fill = fill1
                    for row_index in range(2, worksheet.max_row + 1): # Start from the second row (after header)
                        group_id = worksheet.cell(row=row_index, column=group_column_index).value
                        if group_id != current_group:
                            current_group = group_id
                            current_fill = fill1 if current_fill == fill2 else fill2 # Switch color

                        for col_index in range(1, worksheet.max_column + 1):
                            worksheet.cell(row=row_index, column=col_index).fill = current_fill


            # Автоподбор ширины столбцов
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if cell.value is not None:
                            lines = str(cell.value).split('\n')
                            for line in lines:
                                if len(line) > max_length:
                                    max_length = len(line)
                    except:
                        pass
                adjusted_width = min(max_length + 2, 75) # Increased max width
                worksheet.column_dimensions[column_letter].width = adjusted_width

            # Добавляем автофильтр на лист Статистика и Дубликаты
            if sheet_name in ['Статистика', 'Дубликаты', 'Уникальные', 'Детальный анализ дубликатов']: # Added detailed analysis sheet
                worksheet.auto_filter.ref = worksheet.dimensions

    except Exception as e:
        logging.error(f"Ошибка при форматировании Excel: {e}")

def generate_report_final(original_df, unique_df, duplicates_detailed, work_df, column_mapping):
    """Creation of the final report with enhanced statistics and detailed duplicate analysis."""
    try:
        print("\nЭтап 4: Формирование итогового отчета...")

        output_path = Path('отчет_обращений_финал.xlsx')

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. Sheet: Исходные данные
            original_df.to_excel(writer, sheet_name='Исходные данные', index=False)
            print("✓ Лист 'Исходные данные' создан")

            # 2. Sheet: Уникальные обращения
            if len(unique_df) > 0:
                unique_df.to_excel(writer, sheet_name='Уникальные', index=False)
                print("✓ Лист 'Уникальные' создан")
            else:
                pd.DataFrame({'Сообщение': ['Уникальные обращения не найдены']}).to_excel(
                    writer, sheet_name='Уникальные', index=False
                )

            # 3. Sheet: Дубликаты (Detailed)
            if len(duplicates_detailed) > 0:
                duplicates_detailed.to_excel(writer, sheet_name='Дубликаты', index=False)
                print("✓ Лист 'Дубликаты' создан")
            else:
                pd.DataFrame({'Сообщение': ['Дубликаты не найдены']}).to_excel(
                    writer, sheet_name='Дубликаты', index=False
                )

            # 4. Sheet: Статистика (Enhanced)
            stats_df = create_detailed_statistics_enhanced(original_df, unique_df, duplicates_detailed, work_df)
            stats_df.to_excel(writer, sheet_name='Статистика', index=False)
            print("✓ Лист 'Статистика' создан")

            # 5. Sheet: Графики анализа (Enhanced)
            create_analysis_charts_enhanced(work_df, duplicates_detailed, writer)
            print("✓ Лист 'Графики анализа' создан")

            # 6. Sheet: Детальный анализ дубликатов
            if len(duplicates_detailed) > 0:
                print("Создание листа 'Детальный анализ дубликатов'...")
                detailed_analysis_sheet_name = 'Детальный анализ дубликатов'
                workbook = writer.book
                # Check if the sheet already exists and remove it if it does
                if detailed_analysis_sheet_name in workbook.sheetnames:
                    workbook.remove(workbook[detailed_analysis_sheet_name])
                analysis_worksheet = workbook.create_sheet(detailed_analysis_sheet_name)

                # Write headers
                headers = ['Группа дубликатов', 'Количество записей в группе', 'Индекс записи', 'Тип записи в группе', 'Критерий совпадения', 'Суммарный балл совпадения']
                analysis_worksheet.append(headers)

                # Group work_df by 'Группа дубликатов' to get all records in each group (primary + duplicates)
                # Filter work_df to only include records that are part of a duplicate group
                duplicate_group_ids = duplicates_detailed['Группа дубликатов'].unique()
                # Ensure that group_ids are not empty strings before grouping
                valid_duplicate_group_ids = [gid for gid in duplicate_group_ids if gid != '']
                grouped_work_df = work_df[work_df['Группа дубликатов'].isin(valid_duplicate_group_ids)].groupby('Группа дубликатов')


                for group_id, group_data in grouped_work_df:
                    group_size = len(group_data)
                    # primary_record = group_data[group_data['Тип записи'] == 'Основной в группе']
                    # primary_index = primary_record.index[0] if not primary_record.empty else 'N/A'

                    # Sort records within the group to show primary first
                    group_data_sorted = group_data.sort_values(by='Тип записи', ascending=False) # 'Основной в группе' will be first

                    for index, row in group_data_sorted.iterrows():
                        analysis_worksheet.append([
                            group_id,
                            group_size,
                            index, # Original index
                            row['Тип записи'],
                            row['Критерий совпадения'],
                            row['Суммарный балл совпадения']
                        ])
                    # Add a blank row between groups for readability
                    analysis_worksheet.append([])

                print("✓ Лист 'Детальный анализ дубликатов' создан")
            else:
                 pd.DataFrame({'Сообщение': ['Дубликаты не найдены, детальный анализ не требуется']}).to_excel(
                    writer, sheet_name='Детальный анализ дубликатов', index=False
                )


            format_excel_report(writer)

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

        # Add file download here
        files.download(output_path)
        print(f"\n⬇️ Файл '{output_path.name}' загружен.")


        return True

    except Exception as e:
        logging.error(f"Ошибка при создании отчета: {e}")
        print(f"\n❌ Ошибка при формировании отчета: {e}")
        return False


# --- Define default parameters (can be adjusted by the user) ---
DEFAULT_TFIDF_THRESHOLD = 0.4
DEFAULT_NAME_FUZZY_THRESHOLD = 80
DEFAULT_ADDRESS_FUZZY_THRESHOLD = 75
DEFAULT_RES_FUZZY_THRESHOLD = 85
DEFAULT_FILIAL_FUZZY_THRESHOLD = 85
DEFAULT_CRITERIA_WEIGHTS = {
    'телефоны': 10,
    'текст (TF-IDF)': 8,
    'потребитель (fuzzy)': 7,
    'адрес (fuzzy)': 6,
    'тип обращения (exact)': 5,
    'тематика обращения (exact)': 5,
    'дата': 4,
    'РЭС (fuzzy/exact)': 3,
    'Филиал (fuzzy/exact)': 2
}
DEFAULT_SCORE_THRESHOLD = 15 # This threshold needs tuning based on desired precision/recall

# --- Main function with adjustable parameters and final report ---
def main_final(
    tfidf_threshold=DEFAULT_TFIDF_THRESHOLD,
    name_fuzzy_threshold=DEFAULT_NAME_FUZZY_THRESHOLD,
    address_fuzzy_threshold=DEFAULT_ADDRESS_FUZZY_THRESHOLD,
    res_fuzzy_threshold=DEFAULT_RES_FUZZY_THRESHOLD,
    filial_fuzzy_threshold=DEFAULT_FILIAL_FUZZY_THRESHOLD,
    score_threshold=DEFAULT_SCORE_THRESHOLD,
    criteria_weights=DEFAULT_CRITERIA_WEIGHTS
):
    """
    Main function for consumer appeal analysis with adjustable duplicate detection parameters
    and final detailed report generation.

    Parameters:
    -----------
    tfidf_threshold : float
        Threshold for TF-IDF cosine similarity for text comparison (0.0 to 1.0).
    name_fuzzy_threshold : int
        Fuzzy similarity threshold (0 to 100) for matching consumer names.
    address_fuzzy_threshold : int
        Fuzzy similarity threshold (0 to 100) for matching addresses (original and extracted).
    res_fuzzy_threshold : int
        Fuzzy similarity threshold (0 to 100) for matching RЭС names.
    filial_fuzzy_threshold : int
        Fuzzy similarity threshold (0 to 100) for matching Филиал names.
    score_threshold : float
        Minimum total weighted score for two records to be considered a duplicate.
    criteria_weights : dict
        Dictionary mapping criteria names to their corresponding weights.
    """
    try:
        print("=" * 60)
        print("📈 УЛУЧШЕННЫЙ АНАЛИЗ ОБРАЩЕНИЙ ПОТРЕБИТЕЛЕЙ (ФИНАЛЬНАЯ ВЕРСИЯ)")
        print("=" * 60)

        # Print current parameter values
        print("\nИспользуемые параметры:")
        print(f"  Порог TF-IDF для текста: {tfidf_threshold}")
        print(f"  Порог Fuzzy для Потребителя: {name_fuzzy_threshold}")
        print(f"  Порог Fuzzy для Адреса: {address_fuzzy_threshold}")
        print(f"  Порог Fuzzy для РЭС: {res_fuzzy_threshold}")
        print(f"  Порог Fuzzy для Филиала: {filial_fuzzy_threshold}")
        print(f"  Порог суммарного балла для дубликата: {score_threshold}")
        print(f"  Веса критериев: {criteria_weights}")
        print("-" * 30)


        print("\n🗂️  Загрузите файл Excel с обращениями.")
        file_path = get_file_path()

        if not file_path:
            return

        df = load_data(file_path)
        if df is None:
            print("❌ Не удалось загрузить данные. Программа завершена.")
            return

        df, column_mapping = validate_data(df)
        if df is None:
            print("❌ Данные не прошли проверку. Программа завершена.")
            return

        # Use the weighted duplicate detection function, passing parameters
        unique_df, duplicates_detailed, work_df = find_unique_and_duplicates_weighted(
            df,
            column_mapping,
            tfidf_threshold=tfidf_threshold,
            name_fuzzy_threshold=name_fuzzy_threshold,
            address_fuzzy_threshold=address_fuzzy_threshold,
            res_fuzzy_threshold=res_fuzzy_threshold,
            filial_fuzzy_threshold=filial_fuzzy_threshold,
            score_threshold=score_threshold,
            criteria_weights=criteria_weights
        )

        # Use the final generate_report function
        success = generate_report_final(df, unique_df, duplicates_detailed, work_df, column_mapping)

        if success:
            print("\n" + "=" * 60)
            print("✅ ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
            print("=" * 60)
            print(f"📈 ИТОГОВАЯ СТАТИСТИКА:")
            print(f"   • Исходные записи: {len(df)}")
            print(f"   • Уникальные записи: {len(unique_df)}")
            print(f"   • Дубликаты: {len(duplicates_detailed)}")

        else:
            print("\n❌ Ошибка при создании отчета.")

    except Exception as e:
        logging.error(f"Критическая ошибка в main_final: {e}")
        print(f"\n❌ Критическая ошибка: {e}")

# --- How to adjust parameters ---
# To run the analysis with different parameters, call main_final
# and pass the desired values. For example:
#
# main_final(
#     tfidf_threshold=0.5,           # Increase text similarity requirement
#     name_fuzzy_threshold=90,       # Increase name match strictness
#     score_threshold=20,            # Increase total score requirement
#     criteria_weights={             # Modify weights (example: increase phone weight)
#         'телефоны': 15,
#         'текст (TF-IDF)': 8,
#         'потребитель (fuzzy)': 7,
#         'адрес (fuzzy)': 6,
#         'тип обращения (exact)': 5,
#         'тематика обращения (exact)': 5,
#         'дата': 4,
#         'РЭС (fuzzy/exact)': 3,
#         'Филиал (fuzzy/exact)': 2
#     }
# )
#
# To run with default parameters, simply call:
# main_final()

# Call the main function with default parameters to demonstrate execution
main_final()