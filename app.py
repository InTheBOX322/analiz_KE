import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

from analiz import (
    load_data,
    validate_data,
    generate_report_final,
    find_unique_and_duplicates_weighted,
    create_detailed_statistics_enhanced
)

# Пути для файлов
PROCESS_DIR = Path("process_data")
OUTPUT_PATH = Path("отчет_обращений_финал.xlsx")
PROCESS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Анализ обращений", layout="wide")

st.title("📊 Веб-сервис анализа обращений")
st.write("Загрузите Excel-файл, получите аналитику и скачайте готовый отчёт.")

# --- Загрузка файла ---
uploaded_file = st.file_uploader("Выберите Excel-файл", type=["xlsx", "xls"])

if uploaded_file is not None:
    input_path = PROCESS_DIR / uploaded_file.name
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Файл сохранён: {input_path}")

    # --- Загрузка данных ---
    df = load_data(str(input_path))
    if df is not None:
        st.subheader("📑 Предпросмотр данных")
        st.dataframe(df.head(20))

        if st.button("▶️ Запустить обработку"):
            try:
                # --- Валидация ---
                df_valid, column_mapping = validate_data(df)

                # --- Определение уникальных и дубликатов ---
                unique_df, duplicates_detailed, work_df = find_unique_and_duplicates_weighted(df_valid, column_mapping)

                # --- Формирование Excel отчёта ---
                generate_report_final(df_valid, unique_df, duplicates_detailed, work_df, column_mapping)

                # --- Основная статистика ---
                total = len(df_valid)
                uniq = len(unique_df)
                dups = len(duplicates_detailed)
                st.subheader("📈 Основные показатели")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Всего обращений", total)
                col2.metric("Уникальных", uniq)
                col3.metric("Дубликатов", dups)
                col4.metric("Доля дубликатов", f"{dups / total * 100:.1f}%" if total else "0%")

                # --- Детальная статистика ---
                stats_df = create_detailed_statistics_enhanced(df_valid, unique_df, duplicates_detailed, work_df)
                if not stats_df.empty:
                    st.subheader("📊 Детальная статистика")
                    st.dataframe(stats_df)

                # --- Визуализация ---
                st.subheader("📉 Графики анализа")

                # Категории потребителей
                if "Категория потребителя" in work_df.columns:
                    fig = px.pie(
                        work_df,
                        names="Категория потребителя",
                        title="Распределение по категориям потребителей"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Дубликаты по группам
                if "Группа дубликатов" in work_df.columns:
                    dup_groups = work_df[work_df["Группа дубликатов"] != ""].groupby("Группа дубликатов").size()
                    if not dup_groups.empty:
                        fig = px.histogram(
                            dup_groups,
                            nbins=20,
                            title="Распределение размеров групп дубликатов"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # По месяцам
                date_col = next((c for c in df_valid.columns if "дата" in c.lower()), None)
                if date_col:
                    df_valid[date_col] = pd.to_datetime(df_valid[date_col], errors="coerce")
                    monthly = df_valid[df_valid[date_col].notna()][date_col].dt.to_period("M").value_counts().sort_index()
                    if not monthly.empty:
                        fig = px.bar(
                            monthly,
                            x=monthly.index.astype(str),
                            y=monthly.values,
                            title="Распределение обращений по месяцам"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # --- Скачивание отчёта ---
                if OUTPUT_PATH.exists():
                    with open(OUTPUT_PATH, "rb") as f:
                        st.download_button(
                            label="⬇️ Скачать обработанный отчёт (Excel)",
                            data=f,
                            file_name=OUTPUT_PATH.name,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    st.success("✅ Обработка завершена. Отчёт готов!")

            except Exception as e:
                st.error(f"Ошибка при обработке: {e}")
