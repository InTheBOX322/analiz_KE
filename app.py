# app.py
"""
Streamlit-приложение: быстрый и точный анализ уникальности / дубликатов обращений.
Оптимизации: кэширование TF-IDF, blocking, NearestNeighbors, union-find группировка.
Сохраняет входные файлы и результаты в ./process_data/
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import datetime
import io
from typing import Tuple, List, Iterable, Dict
import openpyxl

# Ensure stopwords
try:
    stopwords.words('russian')
except LookupError:
    nltk.download('stopwords')

# ---------- Config ----------
PROCESS_DIR = Path("./process_data")
PROCESS_DIR.mkdir(parents=True, exist_ok=True)

PHONE_PATTERN = re.compile(r'(?:\+7|8)?\s*\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}')
# Precompiled simple non-digit removal
NON_DIGIT = re.compile(r'\D')

# ---------- Utilities ----------
def extract_phone_numbers_frozen(text) -> frozenset:
    """Найти телефоны и нормализовать как +7XXXXXXXXXX. Возвращает frozenset для быстрого сравнения."""
    if pd.isna(text):
        return frozenset()
    s = str(text)
    matches = PHONE_PATTERN.findall(s)
    normalized = []
    for m in matches:
        d = NON_DIGIT.sub('', m)
        if len(d) == 11 and d.startswith('8'):
            d = '7' + d[1:]
        if len(d) == 11 and d.startswith('7'):
            normalized.append('+' + d)
        elif len(d) == 10:
            normalized.append('+7' + d)
    return frozenset(normalized)

def detect_key_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Определяем колонки name/text/address по названиям; возвращаем словарь имен колонок (None если не найдено)."""
    name_col = text_col = addr_col = None
    for c in df.columns:
        lc = str(c).lower()
        if not name_col and any(k in lc for k in ['потреб', 'клиент', 'заявител', 'имя', 'фамилия', 'контакт']):
            name_col = c
        if not text_col and any(k in lc for k in ['текст', 'обращ', 'описан', 'жалоб', 'сообщен']):
            text_col = c
        if not addr_col and any(k in lc for k in ['адрес', 'местополож', 'объект', 'улица', 'дом']):
            addr_col = c
    # fallback: первый object столбец для текста
    if text_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break
    return {'name': name_col, 'text': text_col, 'address': addr_col}

# ---------- TF-IDF building & neighbors ----------
@st.cache_data(show_spinner=False)
def build_tfidf(texts: Iterable[str]) -> Tuple[TfidfVectorizer, 'scipy.sparse.csr_matrix']:
    """Построить TF-IDF векторизатор и матрицу (кэшируется Streamlit)."""
    vec = TfidfVectorizer(stop_words=stopwords.words('russian'), max_df=0.95, min_df=1)
    tfidf = vec.fit_transform((str(t).lower() for t in texts))
    return vec, tfidf

def find_tfidf_candidates(tfidf_matrix, top_k=5, similarity_threshold=0.4) -> List[tuple]:
    """
    Для каждой строки находим top_k соседей (по cosine distance через NearestNeighbors) и возвращаем
    список кандидатов (i, j, sim) с sim = cosine similarity.
    """
    n = tfidf_matrix.shape[0]
    if n <= 1:
        return []
    k = min(top_k + 1, n)  # +1 потому что сам сосед
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
    nn.fit(tfidf_matrix)
    distances, indices = nn.kneighbors(tfidf_matrix)
    candidates = []
    for i in range(n):
        for dist, j in zip(distances[i], indices[i]):
            if i >= j:
                continue
            sim = 1.0 - dist  # NearestNeighbors cosine distance = 1 - cos_sim
            if sim >= similarity_threshold:
                candidates.append((i, j, float(sim)))
    return candidates

# ---------- Union-Find (DSU) for grouping ----------
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        else:
            self.parent[rb] = ra
            if self.rank[ra] == self.rank[rb]:
                self.rank[ra] += 1

    def components(self):
        comp = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            comp.setdefault(r, []).append(i)
        return comp

# ---------- Core detection (optimized) ----------
def detect_duplicates_optimized(
    df: pd.DataFrame,
    tfidf_threshold=0.45,
    top_k=5,
    name_fuzzy_threshold=85,
    addr_fuzzy_threshold=75,
    score_threshold=12,
    weights=None,
    block_on_phone=True,
    block_on_name_prefix=True
):
    """
    Основная оптимизированная функция:
    - Предобрабатывает: извлекает телефоны (frozenset), нормализует текст/имя/addr.
    - Строит TF-IDF один раз.
    - Получает кандидатов через TF-IDF + телефонные совпадения.
    - Для кандидатов вычисляет fuzzy-оценки (rapidfuzz) и суммарный балл.
    - Группирует пары через DSU.
    Возвращает: unique_df, duplicates_df, work_df (work_df содержит служебные поля).
    """
    if weights is None:
        weights = {'phones': 15, 'tfidf': 8, 'name': 5, 'address': 4}

    df_work = df.copy().reset_index(drop=True)
    n = len(df_work)
    cols = detect_key_columns(df_work)
    name_col = cols['name']
    text_col = cols['text']
    addr_col = cols['address']

    df_work['_name'] = df_work[name_col].fillna('').astype(str) if name_col else pd.Series(['']*n)
    df_work['_text'] = df_work[text_col].fillna('').astype(str) if text_col else pd.Series(['']*n)
    df_work['_addr'] = df_work[addr_col].fillna('').astype(str) if addr_col else pd.Series(['']*n)

    # phones: извлечь по всей строке и сделать frozenset
    df_work['_phones'] = df_work.apply(lambda r: extract_phone_numbers_frozen(' '.join(map(str, r.values))), axis=1)

    # text list for TF-IDF
    texts = df_work['_text'].fillna('').astype(str).tolist()
    vec, tfidf = build_tfidf(texts)

    # Candidates set
    candidate_pairs = set()

    # 1) TF-IDF-based neighbors
    tfidf_cands = find_tfidf_candidates(tfidf, top_k=top_k, similarity_threshold=tfidf_threshold)
    for i, j, sim in tfidf_cands:
        candidate_pairs.add((i, j))

    # 2) Phone-based exact matches (blocking): если телефоны совпадают — обязательный кандидат
    if block_on_phone:
        phone_map = {}
        for idx, phones in enumerate(df_work['_phones']):
            for ph in phones:
                phone_map.setdefault(ph, []).append(idx)
        for ph, idxs in phone_map.items():
            if len(idxs) > 1:
                # add all pairs within idxs
                for a_idx in range(len(idxs)):
                    for b_idx in range(a_idx + 1, len(idxs)):
                        candidate_pairs.add((idxs[a_idx], idxs[b_idx]))

    # 3) Blocking by name prefix (optional) — сравниваем внутри маленьких блоков
    if block_on_name_prefix:
        name_buckets = {}
        for idx, nm in enumerate(df_work['_name']):
            key = (str(nm).strip()[:2].lower() if nm else '')
            name_buckets.setdefault(key, []).append(idx)
        for key, idxs in name_buckets.items():
            if 1 < len(idxs) <= 50:  # ограничение размера блока
                for a in range(len(idxs)):
                    for b in range(a+1, len(idxs)):
                        candidate_pairs.add((idxs[a], idxs[b]))

    # Evaluate candidate pairs with fuzzy and tfidf similarity (fetch tfidf similarity on demand)
    dsu = DSU(n)
    pair_scores = {}  # (i,j) -> score and criteria
    total_pairs = len(candidate_pairs)
    # progress bar
    prog = st.session_state.get("_prog_obj")
    if prog is None:
        prog = None

    i_pair = 0
    for (i, j) in candidate_pairs:
        i_pair += 1
        # update progress if provided
        if prog:
            prog.progress(min(int(i_pair / max(1, total_pairs) * 100), 100))
        score = 0
        criteria = []

        # phones
        phones_i = df_work.at[i, '_phones']
        phones_j = df_work.at[j, '_phones']
        if phones_i and phones_j and (set(phones_i) & set(phones_j)):
            score += weights['phones']; criteria.append('phone')

        # tfidf similarity (compute quickly via vectors)
        # compute cosine similarity using dot product of normalized TF-IDF rows
        try:
            vec_i = tfidf[i]
            vec_j = tfidf[j]
            # cosine similarity via dot (since sklearn uses normalized tf-idf by default)
            sim = float((vec_i @ vec_j.T).A1[0])  # .A1 to get scalar
        except Exception:
            sim = 0.0
        if sim >= tfidf_threshold:
            score += weights['tfidf']; criteria.append(f'tfidf:{sim:.2f}')

        # fuzzy name
        n1 = df_work.at[i, '_name']
        n2 = df_work.at[j, '_name']
        if n1 and n2:
            name_score = fuzz.token_set_ratio(str(n1), str(n2))
            if name_score >= name_fuzzy_threshold:
                score += weights['name']; criteria.append(f'name:{int(name_score)}')

        # fuzzy address
        a1 = df_work.at[i, '_addr']
        a2 = df_work.at[j, '_addr']
        if a1 and a2:
            addr_score = fuzz.token_sort_ratio(str(a1), str(a2))
            if addr_score >= addr_fuzzy_threshold:
                score += weights['address']; criteria.append(f'addr:{int(addr_score)}')

        if score >= score_threshold:
            # mark as duplicate: union them
            dsu.union(i, j)
            pair_scores[(i, j)] = {'score': score, 'criteria': criteria, 'tfidf': sim}

    # build groups from DSU components
    comps = dsu.components()
    group_map = {}
    group_id = 0
    for root, members in comps.items():
        # if single member and never joined with anyone with score -> it might be unique
        if len(members) == 1:
            # only assign group to singletons if they were in pair_scores (i.e., flagged)
            # but DSU might have left them as single if not unioned — they remain unique
            continue
        group_id += 1
        for m in members:
            group_map[m] = f"G{group_id}"

    # Create work_df with group labels and summary
    df_work['Группа дубликатов'] = df_work.index.map(lambda x: group_map.get(x, ''))
    # Add criteria and score aggregated per row (sum of pairwise scores with others in group)
    df_work['Суммарный балл совпадения'] = 0.0
    df_work['Критерий совпадения'] = ''

    # For each pair that was accepted, annotate rows
    for (i, j), info in pair_scores.items():
        gi = group_map.get(i, '')
        gj = group_map.get(j, '')
        if gi:
            # accumulate on each member
            df_work.at[i, 'Суммарный балл совпадения'] += info['score']
            df_work.at[i, 'Критерий совпадения'] = ', '.join(set(filter(None, [df_work.at[i, 'Критерий совпадения'], *info['criteria']])))
        if gj:
            df_work.at[j, 'Суммарный балл совпадения'] += info['score']
            df_work.at[j, 'Критерий совпадения'] = ', '.join(set(filter(None, [df_work.at[j, 'Критерий совпадения'], *info['criteria']])))

    # duplicates_df: rows that have a group id
    duplicates_df = df_work[df_work['Группа дубликатов'] != ''].copy().reset_index(drop=True)
    unique_df = df_work[df_work['Группа дубликатов'] == ''].copy().reset_index(drop=True)

    # stop progress
    if prog:
        prog.progress(100)

    return unique_df, duplicates_df, df_work

# ---------- Report generation ----------
def generate_report_excel(output_path: Path, original_df: pd.DataFrame, unique_df: pd.DataFrame, duplicates_df: pd.DataFrame):
    try:
        stats = [
            {'Показатель': 'Всего обращений', 'Значение': int(len(original_df))},
            {'Показатель': 'Уникальных обращений', 'Значение': int(len(unique_df))},
            {'Показатель': 'Дубликатов (записей)', 'Значение': int(len(duplicates_df))},
            {'Показатель': 'Доля дубликатов (%)',
             'Значение': f"{(len(duplicates_df)/len(original_df)*100):.1f}%" if len(original_df) else "0%"}
        ]
        stats_df = pd.DataFrame(stats)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            original_df.to_excel(writer, sheet_name='Исходные', index=False)
            unique_df.to_excel(writer, sheet_name='Уникальные', index=False)
            duplicates_df.to_excel(writer, sheet_name='Дубликаты', index=False)
            stats_df.to_excel(writer, sheet_name='Статистика', index=False)
        return True, None
    except Exception as e:
        return False, str(e)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Оптимизированный анализ обращений", layout="wide")
st.title("📊 Оптимизированный анализ обращений — уникальность и дубликаты")
st.markdown(
    """
    Загрузите Excel-файл (.xlsx / .xls). Файл сохраняется в `process_data/`.
    Алгоритм использует TF-IDF + rapidfuzz + phone-blocking + union-find для корректных групп дубликатов.
    """
)

uploaded = st.file_uploader("Выберите Excel-файл", type=["xlsx", "xls"])
if not uploaded:
    st.info("Загрузите файл чтобы начать.")
    st.stop()

# Save uploaded file
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
input_path = PROCESS_DIR / f"{ts}_{uploaded.name}"
with open(input_path, "wb") as f:
    f.write(uploaded.getbuffer())
st.success(f"Файл сохранён: `{input_path}`")

# Try load file
try:
    df = pd.read_excel(input_path)
except Exception as e:
    st.error(f"Ошибка чтения Excel: {e}")
    st.stop()

st.subheader("Предпросмотр данных")
st.dataframe(df.head(10))

# Panel: thresholds
with st.expander("Параметры обнаружения (тонкая настройка)"):
    col1, col2 = st.columns(2)
    with col1:
        tfidf_thr = st.slider("TF-IDF threshold (similarity)", 0.0, 1.0, 0.45, 0.01)
        top_k = st.slider("TF-IDF neighbors (top_k)", 1, 20, 5)
        score_thr = st.slider("Порог суммарного балла", 1, 100, 12)
    with col2:
        name_thr = st.slider("Fuzzy имя (threshold)", 50, 100, 85)
        addr_thr = st.slider("Fuzzy адрес (threshold)", 50, 100, 75)
        phones_block = st.checkbox("Использовать блокировку по телефонам (рекомендуется)", value=True)
        name_block = st.checkbox("Использовать блокировку по префиксу имени", value=True)

# Start processing
if st.button("▶️ Запустить обработку"):
    # create a progress object in session to be used inside function
    prog = st.progress(0)
    st.session_state["_prog_obj"] = prog

    try:
        unique_df, duplicates_df, work_df = detect_duplicates_optimized(
            df,
            tfidf_threshold=tfidf_thr,
            top_k=top_k,
            name_fuzzy_threshold=name_thr,
            addr_fuzzy_threshold=addr_thr,
            score_threshold=score_thr,
            weights={'phones': 15, 'tfidf': 8, 'name': 5, 'address': 4},
            block_on_phone=phones_block,
            block_on_name_prefix=name_block
        )

        # cleanup session progress
        prog.empty()
        if "_prog_obj" in st.session_state:
            del st.session_state["_prog_obj"]

        # Metrics
        total = len(df)
        uniq = len(unique_df)
        dups = len(duplicates_df)
        st.subheader("Ключевые показатели")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Всего обращений", f"{total}")
        c2.metric("Уникальных", f"{uniq}")
        c3.metric("Дубликатов", f"{dups}")
        c4.metric("Доля дубликатов", f"{(dups/total*100):.1f}%" if total else "0%")

        # Show examples
        st.subheader("Примеры уникальных записей")
        st.dataframe(unique_df.head(10))

        st.subheader("Примеры дубликатов (с группами)")
        # show sample with group id and criteria
        display_cols = list(df.columns) + ['Группа дубликатов', 'Критерий совпадения', 'Суммарный балл совпадения']
        st.dataframe(duplicates_df[display_cols].head(50))

        # Charts
        st.subheader("Графики временной аналитики")
        # detect date column
        date_col = next((c for c in df.columns if "дата" in str(c).lower() or "date" in str(c).lower()), None)
        if date_col:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                monthly = df[df[date_col].notna()][date_col].dt.to_period('M').value_counts().sort_index()
                if not monthly.empty:
                    fig_month = px.bar(x=monthly.index.astype(str), y=monthly.values,
                                       labels={"x": "Месяц", "y": "Количество"},
                                       title="Распределение обращений по месяцам")
                    st.plotly_chart(fig_month, use_container_width=True)

                valid = df[df[date_col].notna()].copy()
                valid['dow_en'] = valid[date_col].dt.day_name()
                dow_map = {
                    'Monday': 'Понедельник', 'Tuesday': 'Вторник', 'Wednesday': 'Среда',
                    'Thursday': 'Четверг', 'Friday': 'Пятница', 'Saturday': 'Суббота', 'Sunday': 'Воскресенье'
                }
                valid['День недели'] = valid['dow_en'].map(dow_map)
                day_stats = valid['День недели'].value_counts().reindex(
                    ['Понедельник','Вторник','Среда','Четверг','Пятница','Суббота','Воскресенье']
                ).fillna(0)
                if day_stats.sum() > 0:
                    fig_day = px.bar(x=day_stats.index, y=day_stats.values,
                                     labels={"x": "День недели", "y": "Количество"},
                                     title="Распределение обращений по дням недели")
                    st.plotly_chart(fig_day, use_container_width=True)
            except Exception as e:
                st.warning(f"Не удалось построить временную аналитику: {e}")
        else:
            st.info("Не найден столбец с датой для временной аналитики.")

        # Category pie if any
        cat_col = next((c for c in df.columns if 'катег' in str(c).lower() or 'тип' in str(c).lower()), None)
        if cat_col:
            cat_stats = df[cat_col].fillna('Не указано').value_counts()
            if not cat_stats.empty:
                fig_cat = px.pie(values=cat_stats.values, names=cat_stats.index, title="Распределение по категориям")
                st.plotly_chart(fig_cat, use_container_width=True)

        # Save report and provide download
        out_name = f"отчет_обращений_{ts}.xlsx"
        out_path = PROCESS_DIR / out_name
        ok, err = generate_report_excel(out_path, df, unique_df, duplicates_df)
        if ok:
            st.success(f"Отчёт сформирован: `{out_path}`")
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Скачать отчёт (Excel)", f, file_name=out_path.name,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.error(f"Ошибка при сохранении отчёта: {err}")

    except Exception as e:
        # cleanup progress
        try:
            prog.empty()
        except Exception:
            pass
        if "_prog_obj" in st.session_state:
            del st.session_state["_prog_obj"]
        st.error(f"Ошибка во время обработки: {e}")

