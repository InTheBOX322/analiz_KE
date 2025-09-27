# app.py
"""
Streamlit-приложение для анализа обращений (уникальность / дубликаты).
Сохраняет входные файлы в ./process_data/ и результат в ./process_data/отчет_обращений_<имя>.xlsx

Требования (пример):
pip install streamlit pandas openpyxl scikit-learn nltk fuzzywuzzy python-levenshtein plotly
"""

import streamlit as st
import pandas as pd
import re
from pathlib import Path
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import io
import datetime

# --- Ensure Russian stopwords are available ---
try:
    stopwords.words('russian')
except LookupError:
    nltk.download('stopwords')

# --- Utility functions (extracted and simplified) ---
def extract_phone_numbers(text):
    """Найти номера и нормализовать к формату +7XXXXXXXXXX"""
    if pd.isna(text):
        return []
    text_str = str(text)
    # простой общий паттерн для телефонов
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
        if pd.isna(sim):
            return 0.0
        return float(sim)
    except Exception:
        return 0.0

def fuzzy_similar_names(n1, n2, threshold=80):
    if pd.isna(n1) or pd.isna(n2):
        return False
    try:
        score = fuzz.token_set_ratio(str(n1), str(n2))
        return score >= threshold
    except Exception:
        return False

def fuzzy_similar_addresses(a1, a2, threshold=75):
    if pd.isna(a1) or pd.isna(a2):
        return False
    try:
        score = fuzz.token_sort_ratio(str(a1), str(a2))
        return score >= threshold
    except Exception:
        return False

# --- Data loading & validation ---
def load_data(file_path: str) -> pd.DataFrame:
    """Загрузить Excel-файл в DataFrame и попытаться привести даты"""
    df = pd.read_excel(file_path)
    # пробуем привести потенциальный столбец даты
    for col in df.columns:
        if any(k in str(col).lower() for k in ['дата', 'date', 'создан', 'created']):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
            break
    return df

def detect_key_columns(df: pd.DataFrame):
    """Попытаться определить столбцы: name, text, address"""
    name_col = None
    text_col = None
    addr_col = None
    for c in df.columns:
        lc = str(c).lower()
        if not name_col and any(k in lc for k in ['потреб', 'клиент', 'заявител', 'имя', 'фамилия', 'контакт']):
            name_col = c
        if not text_col and any(k in lc for k in ['текст', 'обращ', 'описан', 'жалоб', 'сообщени']):
            text_col = c
        if not addr_col and any(k in lc for k in ['адрес', 'местополож', 'объект']):
            addr_col = c
    # fallbacks
    if text_col is None:
        # возьмем первый текстовый столбец
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break
    return {'name': name_col, 'text': text_col, 'address': addr_col}

# --- Core algorithm: поиск уникальных и дубликатов ---
def find_unique_and_duplicates_weighted(df: pd.DataFrame,
                                        tfidf_threshold=0.45,
                                        name_fuzzy_threshold=85,
                                        address_fuzzy_threshold=75,
                                        score_threshold=10,
                                        criteria_weights=None):
    """
    Возвращает unique_df, duplicates_detailed, work_df.
    Алгоритм прост: для каждой записи сравниваем с последующими и формируем группы дубликатов.
    """
    if criteria_weights is None:
        criteria_weights = {
            'phones': 12,
            'text': 8,
            'name': 5,
            'address': 4
        }

    work_df = df.copy().reset_index(drop=True)
    # Определяем ключевые поля (если их нет, создаём пустые)
    cols = detect_key_columns(work_df)
    name_col = cols['name']
    text_col = cols['text']
    address_col = cols['address']

    work_df['__name'] = work_df[name_col] if name_col else ''
    work_df['__text'] = work_df[text_col] if text_col else ''
    work_df['__address'] = work_df[address_col] if address_col else ''

    # phones: извлекаем из всех строк (concat)
    work_df['__phones'] = work_df.apply(lambda r: extract_phone_numbers(' '.join([str(x) for x in r.values if pd.notna(x)])), axis=1)

    work_df['Группа дубликатов'] = ''
    work_df['Критерий совпадения'] = ''
    work_df['Суммарный балл совпадения'] = 0.0

    n = len(work_df)
    used = set()
    group_id = 0

    # простой O(n^2) проход
    for i in range(n):
        if i in used:
            continue
        group_id += 1
        work_df.at[i, 'Группа дубликатов'] = str(group_id)
        base_name = work_df.at[i, '__name']
        base_text = work_df.at[i, '__text']
        base_addr = work_df.at[i, '__address']
        base_phones = work_df.at[i, '__phones']

        for j in range(i+1, n):
            if j in used:
                continue
            score = 0
            criteria = []

            # телефоны
            phones_j = work_df.at[j, '__phones']
            if base_phones and phones_j and set(base_phones) & set(phones_j):
                score += criteria_weights['phones']
                criteria.append('телефон')

            # текст TF-IDF
            if tfidf_similarity(base_text, work_df.at[j, '__text']) >= tfidf_threshold:
                score += criteria_weights['text']
                criteria.append('текст')

            # имя fuzzy
            if fuzzy_similar_names(base_name, work_df.at[j, '__name'], threshold=name_fuzzy_threshold):
                score += criteria_weights['name']
                criteria.append('имя')

            # адрес fuzzy
            if fuzzy_similar_addresses(base_addr, work_df.at[j, '__address'], threshold=address_fuzzy_threshold):
                score += criteria_weights['address']
                criteria.append('адрес')

            if score >= score_threshold:
                used.add(j)
                work_df.at[j, 'Группа дубликатов'] = str(group_id)
                work_df.at[j, 'Критерий совпадения'] = ', '.join(criteria)
                work_df.at[j, 'Суммарный балл совпадения'] = score

    duplicates_detailed = work_df[work_df['Группа дубликатов'] != ''].copy()
    # Пометим уникальные: те, кто оказался единственными в своей группе
    group_sizes = duplicates_detailed['Группа дубликатов'].value_counts()
    singleton_groups = group_sizes[group_sizes == 1].index.tolist()
    # строки с группами singleton (т.е. фактически не дубликаты) надо переместить в уникальные
    duplicates_mask = work_df['Группа дубликатов'].isin(group_sizes.index[group_sizes > 1])
    duplicates_df = work_df[duplicates_mask].copy()
    unique_df = work_df[~duplicates_mask].copy()

    # очистим служебные колонки из выдачи (оставим, но можно удалить)
    return unique_df.reset_index(drop=True), duplicates_df.reset_index(drop=True), work_df.reset_index(drop=True)

# --- Report generation: записать Excel с листами 'Исходные', 'Уникальные', 'Дубликаты', 'Статистика' ---
def generate_report(output_path: Path, original_df: pd.DataFrame, unique_df: pd.DataFrame, duplicates_df: pd.DataFrame):
    try:
        stats = []
        total = len(original_df)
        uniq = len(unique_df)
        dups = len(duplicates_df)
        stats.append({'Показатель': 'Всего обращений', 'Значение': total})
        stats.append({'Показатель': 'Уникальных обращений', 'Значение': uniq})
        stats.append({'Показатель': 'Дубликатов (записей)', 'Значение': dups})
        stats.append({'Показатель': 'Доля дубликатов (%)', 'Значение': f"{(dups/total*100):.1f}%" if total else "0%"})

        stats_df = pd.DataFrame(stats)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            original_df.to_excel(writer, sheet_name='Исходные', index=False)
            unique_df.to_excel(writer, sheet_name='Уникальные', index=False)
            duplicates_df.to_excel(writer, sheet_name='Дубликаты', index=False)
            stats_df.to_excel(writer, sheet_name='Статистика', index=False)
        return True, None
    except Exception as e:
        return False, str(e)

# --- Streamlit UI ---
st.set_page_config(page_title="Анализ обращений", layout="wide")
st.title("📊 Анализ обращений — уникальность и дубликаты")
st.markdown("Загрузите Excel-файл. Файл будет сохранён в `process_data/`. После обработки можно скачать отчёт.")

# ensure folders
PROCESS_DIR = Path("./process_data")
PROCESS_DIR.mkdir(parents=True, exist_ok=True)

uploaded = st.file_uploader("Выберите Excel-файл", type=["xlsx", "xls"])
if uploaded is None:
    st.info("Загрузите файл для начала анализа.")
else:
    # save uploaded file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = PROCESS_DIR / f"{timestamp}_{uploaded.name}"
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Файл сохранён: `{input_path}`")

    # load and preview
    try:
        df = load_data(str(input_path))
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        st.stop()

    st.subheader("Предпросмотр (первые строки)")
    st.dataframe(df.head(10))

    # Allow user to tune thresholds (optional)
    with st.expander("Параметры обнаружения (при необходимости)"):
        tfidf_thr = st.slider("TF-IDF threshold (текст)", 0.0, 1.0, 0.45, 0.01)
        name_thr = st.slider("Fuzzy для имени (0-100)", 50, 100, 85, 1)
        addr_thr = st.slider("Fuzzy для адреса (0-100)", 50, 100, 75, 1)
        score_thr = st.slider("Порог суммарного балла", 1, 50, 10, 1)

    if st.button("▶️ Запустить обработку"):
        with st.spinner("Идёт поиск уникальных и дубликатов..."):
            try:
                unique_df, duplicates_df, work_df = find_unique_and_duplicates_weighted(
                    df,
                    tfidf_threshold=tfidf_thr,
                    name_fuzzy_threshold=name_thr,
                    address_fuzzy_threshold=addr_thr,
                    score_threshold=score_thr
                )

                # stats
                total = len(df)
                uniq = len(unique_df)
                dups = len(duplicates_df)
                st.subheader("Ключевые показатели")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Всего обращений", total)
                c2.metric("Уникальных", uniq)
                c3.metric("Дубликатов", dups)
                c4.metric("Доля дубликатов", f"{(dups/total*100):.1f}%" if total else "0%")

                # show small samples
                st.subheader("Пример уникальных записей")
                st.dataframe(unique_df.head(10))

                st.subheader("Пример дубликатов")
                st.dataframe(duplicates_df.head(20))

                # Charts: по месяцам и по дням недели
                # detect date column
                date_col = next((c for c in df.columns if "дата" in str(c).lower() or "date" in str(c).lower()), None)
                if date_col:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        monthly = df[df[date_col].notna()][date_col].dt.to_period('M').value_counts().sort_index()
                        if not monthly.empty:
                            fig1 = px.bar(x=monthly.index.astype(str), y=monthly.values,
                                          labels={'x': 'Месяц', 'y': 'Количество'},
                                          title="Распределение обращений по месяцам")
                            st.plotly_chart(fig1, use_container_width=True)

                        # days of week
                        valid = df[df[date_col].notna()].copy()
                        # day_name in Russian may require locale; use english names then map
                        valid['__dow_en'] = valid[date_col].dt.day_name()
                        dow_map = {
                            'Monday': 'Понедельник', 'Tuesday': 'Вторник', 'Wednesday': 'Среда',
                            'Thursday': 'Четверг', 'Friday': 'Пятница', 'Saturday': 'Суббота', 'Sunday': 'Воскресенье'
                        }
                        valid['День недели'] = valid['__dow_en'].map(dow_map)
                        day_stats = valid['День недели'].value_counts().reindex(
                            ['Понедельник','Вторник','Среда','Четверг','Пятница','Суббота','Воскресенье']
                        ).fillna(0)
                        if day_stats.sum() > 0:
                            fig2 = px.bar(x=day_stats.index, y=day_stats.values,
                                          labels={'x': 'День недели', 'y': 'Количество'},
                                          title="Распределение обращений по дням недели")
                            st.plotly_chart(fig2, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Не удалось построить временную аналитику: {e}")

                # category visualization if present
                # try find category column
                cat_col = next((c for c in df.columns if 'катег' in str(c).lower() or 'тип' in str(c).lower()), None)
                if cat_col:
                    cat_stats = df[cat_col].fillna('Не указано').value_counts()
                    if not cat_stats.empty:
                        fig3 = px.pie(values=cat_stats.values, names=cat_stats.index, title="Распределение по категориям")
                        st.plotly_chart(fig3, use_container_width=True)

                # Save report
                out_name = f"отчет_обращений_{timestamp}.xlsx"
                out_path = PROCESS_DIR / out_name
                ok, err = generate_report(out_path, df, unique_df, duplicates_df)
                if ok:
                    st.success(f"Отчёт сформирован: `{out_path}`")
                    # provide download
                    with open(out_path, "rb") as f:
                        btn = st.download_button(
                            label="⬇️ Скачать отчёт (Excel)",
                            data=f,
                            file_name=out_path.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.error(f"Не удалось сохранить отчёт: {err}")

            except Exception as e:
                st.error(f"Ошибка при обработке: {e}")
