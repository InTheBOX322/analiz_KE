import streamlit as st
import pandas as pd
import os
from pathlib import Path
import logging
from datetime import datetime
import re
from difflib import SequenceMatcher
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill, Alignment
import numpy as np

# Настройка логирования
logging.basicConfig(
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

def load_data(file_path):
    """Загрузка данных с обработкой ошибок"""
    try:
        df = pd.read_excel(file_path)
        logging.info(f"Успешная загрузка файла: {file_path}")
        logging.info(f"Столбцы в файле: {list(df.columns)}")

        # Автоматическое определение столбца с датой
        date_columns = []
        for col in df.columns:
            if any(keyword in str(col).lower() for keyword in ['дата', 'date', 'создан', 'created']):
                date_columns.append(col)

        if date_columns:
            # Берем первый подходящий столбец
            date_column = date_columns[0]
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        st.error(f"Ошибка при загрузке файла: {e}")
        return None

def validate_data(df):
    """Проверка корректности данных"""
    try:
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

        # Проверка на пустые значения
        key_columns = list(found_columns.values())
        if key_columns:
            empty_count = df[key_columns].isnull().sum()
            if empty_count.any():
                st.warning(f"Найдены пустые значения:\n{empty_count}")

        logging.info("Успешная валидация данных")
        return df, found_columns

    except Exception as e:
        logging.error(f"Ошибка валидации данных: {e}")
        st.error(f"Ошибка валидации: {e}")
        return None, None

def find_date_column(df):
    """Поиск столбца с датой"""
    for col in df.columns:
        if any(keyword in str(col).lower() for keyword in ['дата', 'date', 'создан', 'created']):
            return col
    return None

def find_unique_and_duplicates(df, column_mapping):
    """Улучшенный алгоритм определения уникальности обращений с учетом телефонов и дат"""
    try:
        # Получаем названия столбцов из mapping
        consumer_col = column_mapping.get('потребитель', 'Потребитель')
        appeal_col = column_mapping.get('текст обращения', 'Текст обращения')
        address_col = column_mapping.get('адрес', 'Адрес')

        # Находим столбец с датой
        date_col = find_date_column(df)
        if not date_col:
            st.error("❌ Не найден столбец с датой! Проверка уникальности невозможна.")
            return df, pd.DataFrame(), df

        # Создаем рабочую копию
        work_df = df.copy()

        # Извлекаем адреса и телефоны из текста обращений
        work_df['Извлеченный адрес'] = work_df[appeal_col].apply(extract_address)
        work_df['Телефоны'] = work_df[appeal_col].apply(extract_phone_numbers)
        work_df['Количество телефонов'] = work_df['Телефоны'].apply(len)

        # Добавляем нормализованные данные для сравнения
        work_df['Текст_normalized'] = work_df[appeal_col].astype(str).str.lower().str.strip()
        work_df['Потребитель_normalized'] = work_df[consumer_col].astype(str).str.lower().str.strip()
        work_df['Дата_день'] = work_df[date_col].dt.date
        work_df['Дата_неделя'] = work_df[date_col].dt.to_period('W').apply(lambda r: r.start_time.date())
        work_df['Дата_месяц'] = work_df[date_col].dt.to_period('M').apply(lambda r: r.start_time.date())

        # Определяем категорию потребителя
        work_df['Категория потребителя'] = work_df.apply(
            lambda row: categorize_consumer(row[consumer_col], row[appeal_col]), axis=1
        )

        # Создаем колонку для группировки
        work_df['group_key'] = work_df['Потребитель_normalized'] + '|' + work_df['Дата_неделя'].astype(str)

        # Определяем уникальные записи
        work_df['Тип записи'] = 'Уникальная'
        work_df['Группа дубликатов'] = ''
        work_df['Критерий совпадения'] = ''

        # Группируем по потребителю и неделе
        group_keys = work_df['group_key'].unique()
        total_groups = len(group_keys)

        for i, group_key in enumerate(group_keys):
            group = work_df[work_df['group_key'] == group_key]

            if len(group) > 1:
                group_indices = group.index.tolist()

                for i_idx in range(len(group_indices)):
                    idx_i = group_indices[i_idx]
                    if work_df.loc[idx_i, 'Тип записи'] == 'Дубликат':
                        continue

                    text_i = work_df.loc[idx_i, 'Текст_normalized']
                    address_i = work_df.loc[idx_i, 'Извлеченный адрес']
                    phones_i = work_df.loc[idx_i, 'Телефones']
                    date_i = work_df.loc[idx_i, 'Дата_день']

                    for j_idx in range(i_idx + 1, len(group_indices)):
                        idx_j = group_indices[j_idx]
                        if work_df.loc[idx_j, 'Тип записи'] == 'Дубликат':
                            continue

                        text_j = work_df.loc[idx_j, 'Текст_normalized']
                        address_j = work_df.loc[idx_j, 'Извлеченный адрес']
                        phones_j = work_df.loc[idx_j, 'Телефоны']
                        date_j = work_df.loc[idx_j, 'Дата_день']

                        # Проверяем различные критерии совпадения
                        match_criteria = []

                        # 1. Совпадение телефонов (самый сильный критерий)
                        if similar_phones(phones_i, phones_j):
                            match_criteria.append('телефоны')

                        # 2. Совпадение текста
                        if similar_text(text_i, text_j):
                            match_criteria.append('текст')

                        # 3. Совпадение адреса
                        if similar_text(address_i, address_j, threshold=0.7):
                            match_criteria.append('адрес')

                        # 4. Совпадение даты (в пределах 3 дней)
                        date_diff = abs((date_i - date_j).days)
                        if date_diff <= 3:
                            match_criteria.append('дата')

                        # Если есть хотя бы 2 совпадения или телефоны + что-то еще
                        if (len(match_criteria) >= 2 or
                            ('телефоны' in match_criteria and len(match_criteria) >= 1)):

                            if work_df.loc[idx_i, 'Тип записи'] == 'Уникальная':
                                work_df.loc[idx_i, 'Тип записи'] = 'Основной в группе'
                                work_df.loc[idx_i, 'Группа дубликатов'] = f"Группа_{idx_i}"
                                work_df.loc[idx_i, 'Критерий совпадения'] = ', '.join(match_criteria)

                            work_df.loc[idx_j, 'Тип записи'] = 'Дубликат'
                            work_df.loc[idx_j, 'Группа дубликатов'] = f"Группа_{idx_i}"
                            work_df.loc[idx_j, 'Критерий совпадения'] = ', '.join(match_criteria)

        # Создаем финальные DataFrames
        unique_df = work_df[work_df['Тип записи'].isin(['Уникальная', 'Основной в группе'])].copy()
        duplicates_detailed = work_df[work_df['Тип записи'] == 'Дубликат'].copy()

        # Убираем технические колонки из финальных таблиц
        columns_to_drop = ['Текст_normalized', 'Потребитель_normalized', 'Дата_день',
                          'Дата_неделя', 'Дата_месяц', 'group_key', 'Извлеченный адрес', 'Телефоны']

        final_columns = [col for col in unique_df.columns if col not in columns_to_drop]
        unique_df = unique_df[final_columns]
        duplicates_detailed = duplicates_detailed[final_columns]

        logging.info(f"Уникальных обращений: {len(unique_df)}, Дубликатов: {len(duplicates_detailed)}")
        return unique_df, duplicates_detailed, work_df

    except Exception as e:
        logging.error(f"Ошибка при определении уникальности: {e}")
        st.error(f"❌ Ошибка при обработке данных: {e}")
        return df, pd.DataFrame(), df

def create_detailed_statistics(original_df, unique_df, duplicates_detailed, work_df):
    """Создание расширенной статистики с графиками"""
    try:
        stats_data = []

        # Основная статистика
        total_appeals = len(original_df)
        total_unique = len(unique_df)
        total_duplicates = len(duplicates_detailed)

        stats_data.append({'Категория': 'ОБЩАЯ СТАТИСТИКА', 'Показатель': 'Всего обращений', 'Значение': total_appeals})
        stats_data.append({'Категория': 'ОБЩАЯ СТАТИСТИКА', 'Показатель': 'Уникальных обращений', 'Значение': total_unique})
        stats_data.append({'Категория': 'ОБЩАЯ СТАТИСТИКА', 'Показатель': 'Дубликатов', 'Значение': total_duplicates})
        stats_data.append({'Категория': 'ОБЩАЯ СТАТИСТИКА', 'Показатель': 'Процент дубликатов',
                          'Значение': f"{(total_duplicates/total_appeals*100):.1f}%" if total_appeals > 0 else '0%'})

        # Статистика по категориям потребителей
        if 'Категория потребителя' in work_df.columns:
            category_stats = work_df['Категория потребителя'].value_counts()
            for category, count in category_stats.items():
                stats_data.append({
                    'Категория': 'КАТЕГОРИИ ПОТРЕБИТЕЛЕЙ',
                    'Показатель': category,
                    'Значение': count
                })

        # Статистика по типам записей
        if 'Тип записи' in work_df.columns:
            type_stats = work_df['Тип записи'].value_counts()
            for type_name, count in type_stats.items():
                stats_data.append({
                    'Категория': 'ТИПЫ ЗАПИСЕЙ',
                    'Показатель': type_name,
                    'Значение': count
                })

        # Статистика по критериям совпадения
        if 'Критерий совпадения' in duplicates_detailed.columns:
            criteria_stats = duplicates_detailed['Критерий совпадения'].value_counts().head(10)
            for criteria, count in criteria_stats.items():
                if criteria:  # Пропускаем пустые
                    stats_data.append({
                        'Категория': 'КРИТЕРИИ СОВПАДЕНИЯ ДУБЛИКАТОВ',
                        'Показатель': criteria,
                        'Значение': count
                    })

        # Статистика по датам
        date_col = find_date_column(original_df)
        if date_col:
            # По месяцам
            original_df['Месяц'] = original_df[date_col].dt.to_period('M')
            monthly_stats = original_df['Месяц'].value_counts().sort_index()
            for month, count in monthly_stats.items():
                stats_data.append({
                    'Категория': 'СТАТИСТИКА ПО МЕСЯЦАМ',
                    'Показатель': f'{month}',
                    'Значение': count
                })

            # По дням недели
            original_df['День недели'] = original_df[date_col].dt.day_name()
            day_stats = original_df['День недели'].value_counts()
            for day, count in day_stats.items():
                stats_data.append({
                    'Категория': 'СТАТИСТИКА ПО ДНЯМ НЕДЕЛИ',
                    'Показатель': day,
                    'Значение': count
                })

        # Статистика по телефонам
        if 'Количество телефонов' in work_df.columns:
            phone_stats = work_df['Количество телефонов'].value_counts().sort_index()
            for count_phones, records in phone_stats.items():
                stats_data.append({
                    'Категория': 'СТАТИСТИКА ПО ТЕЛЕФОНАМ',
                    'Показатель': f'Обращений с {count_phones} телефоном(ами)',
                    'Значение': records
                })

        stats_df = pd.DataFrame(stats_data)
        return stats_df

    except Exception as e:
        logging.error(f"Ошибка при создании статистики: {e}")
        st.error(f"❌ Ошибка при создании статистики: {e}")
        return pd.DataFrame()

def format_excel_report(writer):
    """Форматирование Excel отчета"""
    try:
        workbook = writer.book

        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        center_align = Alignment(horizontal="center", vertical="center")

        for sheet_name in workbook.sheetnames:
            worksheet = workbook[sheet_name]

            # Форматирование заголовков
            for cell in worksheet[1]:
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = center_align

            # Автоподбор ширины столбцов
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

            # Добавляем автофильтр на лист Статистика
            if sheet_name == 'Статистика':
                worksheet.auto_filter.ref = worksheet.dimensions

    except Exception as e:
        logging.error(f"Ошибка при форматировании Excel: {e}")

def generate_report(original_df, unique_df, duplicates_detailed, work_df, column_mapping):
    """Создание итогового отчета с улучшенной статистикой"""
    try:
        # Создаем временную директорию если не существует
        if not os.path.exists("temp"):
            os.makedirs("temp")
            
        output_path = Path("temp/отчет_обращений.xlsx")

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. Лист: Исходные данные
            original_df.to_excel(writer, sheet_name='Исходные данные', index=False)

            # 2. Лист: Уникальные обращения
            if len(unique_df) > 0:
                unique_df.to_excel(writer, sheet_name='Уникальные', index=False)
            else:
                pd.DataFrame({'Сообщение': ['Уникальные обращения не найдены']}).to_excel(
                    writer, sheet_name='Уникальные', index=False
                )

            # 3. Лист: Дубликаты
            if len(duplicates_detailed) > 0:
                duplicates_detailed.to_excel(writer, sheet_name='Дубликаты', index=False)
            else:
                pd.DataFrame({'Сообщение': ['Дубликаты не найдены']}).to_excel(
                    writer, sheet_name='Дубликаты', index=False
                )

            # 4. Лист: Статистика
            stats_df = create_detailed_statistics(original_df, unique_df, duplicates_detailed, work_df)
            if not stats_df.empty:
                stats_df.to_excel(writer, sheet_name='Статистика', index=False)

            format_excel_report(writer)

        logging.info(f"Успешное создание отчета: {output_path}")
        return True

    except Exception as e:
        logging.error(f"Ошибка при создании отчета: {e}")
        st.error(f"❌ Ошибка при формировании отчета: {e}")
        return False

def main():
    st.set_page_config(
        page_title="Анализ обращений потребителей",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 УЛУЧШЕННЫЙ АНАЛИЗ ОБРАЩЕНИЙ ПОТРЕБИТЕЛЕЙ")
    st.write("Загрузите файл Excel для анализа и выявления дубликатов обращений")
    
    # Временная директория для обработки файлов
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    uploaded_file = st.file_uploader("Выберите файл Excel", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # Сохраняем загруженный файл во временную директорию
        file_path = Path("temp") / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Файл '{uploaded_file.name}' успешно загружен!")
        
        if st.button("Начать обработку", type="primary"):
            with st.spinner("Обработка данных..."):
                # Загрузка данных
                df = load_data(file_path)
                
                if df is not None:
                    st.info(f"Данные успешно загружены! Размер данных: {df.shape}")
                    
                    # Валидация данных и определение столбцов
                    df, column_mapping = validate_data(df)
                    
                    if df is not None and column_mapping:
                        st.info("Валидация данных завершена успешно!")
                        
                        # Определение уникальности и дубликатов
                        unique_df, duplicates_detailed, work_df = find_unique_and_duplicates(df, column_mapping)
                        
                        # Генерация отчета во временную директорию
                        success = generate_report(df, unique_df, duplicates_detailed, work_df, column_mapping)
                        
                        if success:
                            st.success("✅ Обработка завершена успешно!")
                            
                            # Отображение статистики
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Всего обращений", len(df))
                            with col2:
                                st.metric("Уникальных обращений", len(unique_df))
                            with col3:
                                st.metric("Дубликатов", len(duplicates_detailed))
                            
                            # Кнопка для скачивания отчета
                            output_path = Path("temp/отчет_обращений.xlsx")
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    label="📥 Скачать отчет",
                                    data=f,
                                    file_name="отчет_обращений.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheet.sheet"
                                )
                            
                            # Очистка временных файлов
                            try:
                                os.remove(file_path)
                                os.remove(output_path)
                            except:
                                pass
                                
                        else:
                            st.error("❌ Произошла ошибка при формировании отчета.")
                    else:
                        st.error("❌ Не удалось определить необходимые столбцы или данные не прошли валидацию.")
                else:
                    st.error("❌ Не удалось загрузить данные из файла.")

if __name__ == "__main__":
    main()