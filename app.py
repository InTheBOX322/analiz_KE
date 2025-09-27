# app.py
"""
Streamlit-приложение для анализа обращений (уникальность / дубликаты).
Сохраняет входные файлы в ./process_data/ и результат в ./process_data/отчет_обращений_<имя>.xlsx

Требования (пример):
pip install streamlit pandas openpyxl scikit-learn nltk rapidfuzz plotly
"""

import streamlit as st
import pandas as pd
import re
from pathlib import Path
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import datetime

# --- Ensure Russian stopwords are available ---
try:
    stopwords.words('russian')
except LookupError:
    nltk.download('stopwords')

# --- Utility functions ---
def extract_phone_numbers(text):
    """Найти номера и нормализовать к формату +7XXXXXXXXXX"""
    if pd.isna(text):
        return []
    text_str = str(text)
    matches = re.findall(r'(?:\+7|8)?\s*\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}', text_str)
    normalized = []
    for m in matches:
        digits = re.sub(r'\D', '', m)
        if len(digits) == 11 and digits.startswith('8'):
            digits = '7' + digits[1:]
        if len(digits) == 11 and digits.startswith('7'):
            normalized.append('+' + digits)
        elif len(digits) == 10:
            normalized.append('+7' + digits)
    return list(set(normalized))

def tfidf_similarity(text1, text2):
    """TF-IDF cosine similarity (0..1)"""
    if pd.isna(text1) or pd.isna(text2):
        return 0.0
    s1 = str(text1).lower()
    s2 = str(text2).lower()
    try:
        vec = TfidfVectorizer(stop_words=stopwords.words('russian'))
        tfidf = vec.fit_transform([s1, s2])
        sim = cosine_similarity(tfidf)[0, 1]
        return float(sim) if not pd.isna(sim) else 0.0
    except Exception:
        return 0.0

def fuzzy_similar_names(n1, n2, threshold=80):
    if pd.isna(n1) or pd.isna(n2):
        return False
    score = fuzz.token_set_ratio(str(n1), str(n2))
    return score >= threshold

def fuzzy_similar_addresses(a1, a2, threshold=75):
    if pd.isna(a1) or pd.isna(a2):
        return False
    score = fuzz.token_sort_ratio(str(a1), str(a2))
    return score >= threshold

# --- Data loading & validation ---
def load_data(file_path: str) -> pd.DataFrame:
    """Загрузить Excel-файл"""
    df = pd.read_excel(file_path)
    # пробуем привести столбцы дат
    for col in df.columns:
        if any(k in str(col).lower() for k in ['дата', 'date', 'создан', 'created']):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            break
    return df

def detect_key_columns(df: pd.DataFrame):
    """Определить name/text/address"""
    name_col, text_col, addr_col = None, None, None
    for c in df.columns:
        lc = str(c).lower()
        if not name_col and any(k in lc for k in ['потреб', 'клиент', 'заявител', 'имя']):
            name_col = c
        if not text_col and any(k in lc for k in ['текст', 'обращ', 'описан', 'жалоб', 'сообщен']):
            text_col = c
        if not addr_col and any(k in lc for k in ['адрес', 'местополож', 'объект']):
            addr_col = c
    if text_col is None:  # fallback
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break
    return {'name': name_col, 'text': text_col, 'address': addr_col}

# --- Core algorithm ---
def find_unique_and_duplicates_weighted(df: pd.DataFrame,
                                        tfidf_threshold=0.45,
                                        name_fuzzy_threshold=85,
                                        address_fuzzy_threshold=75,
                                        score_threshold=10,
                                        criteria_weights=None):
    if criteria_weights is None:
        criteria_weights = {'phones': 12, 'text': 8, 'name': 5, 'address': 4}

    work_df = df.copy().reset_index(drop=True)
    cols = detect_key_columns(work_df)
    work_df['__name'] = work_df[cols['name']] if cols['name'] else ''
    work_df['__text'] = work_df[cols['text']] if cols['text'] else ''
    work_df['__address'] = work_df[cols['address']] if cols['address'] else ''
    work_df['__phones'] = work_df.apply(lambda r: extract_phone_numbers(' '.join([str(x) for x in r.values if pd.notna(x)])), axis=1)

    work_df['Группа дубликатов'] = ''
    work_df['Критерий совпадения'] = ''
    work_df['Суммарный балл совпадения'] = 0.0

    used = set()
    group_id = 0
    n = len(work_df)

    for i in range(n):
        if i in used:
            continue
        group_id += 1
        work_df.at[i, 'Группа дубликатов'] = str(group_id)
        for j in range(i+1, n):
            if j in used:
                continue
            score, criteria = 0, []
            if set(work_df.at[i, '__phones']) & set(work_df.at[j, '__phones']):
                score += criteria_weights['phones']; criteria.append('телефон')
            if tfidf_similarity(work_df.at[i, '__text'], work_df.at[j, '__text']) >= tfidf_threshold:
                score += criteria_weights['text']; criteria.append('текст')
            if fuzzy_similar_names(work_df.at[i, '__name'], work_df.at[j, '__name'], name_fuzzy_threshold):
                score += criteria_weights['name']; criteria.append('имя')
            if fuzzy_similar_addresses(work_df.at[i, '__address'], work_df.at[j, '__address'], address_fuzzy_threshold):
                score += criteria_weights['address']; criteria.append('адрес')
            if score >= score_threshold:
                used.add(j)
                work_df.at[j, 'Группа дубликатов'] = str(group_id)
                work_df.at[j, 'Критерий совпадения'] = ', '.join(criteria)
                work_df.at[j, 'Суммарный балл совпадения'] = score

    group_sizes = work_df['Группа дубликатов'].value_counts()
    duplicates_df = work_df[work_df['Группа дубликатов'].isin(group_sizes.index[group_sizes > 1])]
    unique_df = work_df[~work_df.index.isin(duplicates_df.index)]
    return unique_df, duplicates_df, work_df

def generate_report(output_path: Path, original_df, unique_df, duplicates_df):
    stats = [
        {'Показатель': 'Всего обращений', 'Значение': len(original_df)},
        {'Показатель': 'Уникальных обращений', 'Значение': len(unique_df)},
        {'Показатель': 'Дубликатов (записей)', 'Значение': len(duplicates_df)},
        {'Показатель': 'Доля дубликатов (%)',
         'Значение': f"{(len(duplicates_df)/len(original_df)*100):.1f}%" if len(original_df) else "0%"}
    ]
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        original_df.to_excel(writer, sheet_name="Исходные", index=False)
        unique_df.to_excel(writer, sheet_name="Уникальные", index=False)
        duplicates_df.to_excel(writer, sheet_name="Дубликаты", index=False)
        pd.DataFrame(stats).to_excel(writer, sheet_name="Статистика", index=False)

# --- Streamlit UI ---
st.set_page_config(page_title="Анализ обращений", layout="wide")
st.title("📊 Анализ обращений — уникальность и дубликаты")

PROCESS_DIR = Path("./process_data")
PROCESS_DIR.mkdir(parents=True, exist_ok=True)

uploaded = st.file_uploader("Выберите Excel-файл", type=["xlsx", "xls"])
if uploaded:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = PROCESS_DIR / f"{ts}_{uploaded.name}"
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Файл сохранён: {input_path}")

    df = load_data(str(input_path))
    st.subheader("Предпросмотр")
    st.dataframe(df.head(10))

    if st.button("▶️ Запустить обработку"):
        with st.spinner("Обработка..."):
            unique_df, duplicates_df, work_df = find_unique_and_duplicates_weighted(df)
            total, uniq, dups = len(df), len(unique_df), len(duplicates_df)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Всего", total)
            col2.metric("Уникальные", uniq)
            col3.metric("Дубликаты", dups)
            col4.metric("Доля дубликатов", f"{dups/total*100:.1f}%" if total else "0%")

            st.subheader("Примеры уникальных")
            st.dataframe(unique_df.head(10))
            st.subheader("Примеры дубликатов")
            st.dataframe(duplicates_df.head(10))

            # --- графики по месяцам и дням недели ---
            date_col = next((c for c in df.columns if "дата" in c.lower() or "date" in c.lower()), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                monthly = df[df[date_col].notna()][date_col].dt.to_period("M").value_counts().sort_index()
                if not monthly.empty:
                    st.plotly_chart(px.bar(x=monthly.index.astype(str), y=monthly.values,
                                           labels={"x":"Месяц","y":"Кол-во"},
                                           title="Распределение по месяцам"), use_container_width=True)
                dow_map = {"Monday":"Понедельник","Tuesday":"Вторник","Wednesday":"Среда",
                           "Thursday":"Четверг","Friday":"Пятница","Saturday":"Суббота","Sunday":"Воскресенье"}
                df["День недели"] = df[date_col].dt.day_name().map(dow_map)
                day_stats = df["День недели"].value_counts().reindex(
                    ["Понедельник","Вторник","Среда","Четверг","Пятница","Суббота","Воскресенье"]
                ).fillna(0)
                st.plotly_chart(px.bar(x=day_stats.index, y=day_stats.values,
                                       labels={"x":"День недели","y":"Кол-во"},
                                       title="Распределение по дням недели"), use_container_width=True)

            # --- отчёт ---
            out_path = PROCESS_DIR / f"отчет_обращений_{ts}.xlsx"
            generate_report(out_path, df, unique_df, duplicates_df)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Скачать отчёт (Excel)", f, file_name=out_path.name,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
