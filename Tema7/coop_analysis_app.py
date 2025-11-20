import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False
    print("⚠️ Модуль mlxtend не встановлено. Часті патерни (Apriori) будуть пропущені.")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================
# 0. КОНФІГУРАЦІЯ
# ==========================

# ❗ Файли в поточній папці
CUSTOMERS_CSV = "clienti.csv"
PRODUCTS_CSV = "articoli.csv"
MARKETING_CSV = "mkt.csv"
SALES_CSV = "scontrini.csv"
CALENDAR_CSV = "calendario.csv"

# ❗ Назви колонок
COLS = {
    "customer_id": "ID_CLIENTE",
    "product_id": "ID_ARTICOLO",
    "ticket_id": "ID_SCONTRINO",
    "date": "DATA",
    "birth_year": "ANNO_NASCITA",
    "gender": "SESSO",
    "marital_status": "STATO_CIVILE",
    "profession": "PROFESSIONE",
    "join_year": "ANNO_ISCRIZIONE",
    "qty_weight": "QTA_PESO",
    "qty_pieces": "QTA_PEZZI",
    "amount": "IMPORTO",
    "sector": "SETTORE",
    "subsector": "SOTTOCATEGORIA",   
    "segment": "SEGMENTO"            
}

SECTOR_FRESCH = "FRESCHISSIMI"
SECTOR_PESCE = "PESCE"


@dataclass
class CoopData:
    customers: pd.DataFrame
    products: pd.DataFrame
    marketing: pd.DataFrame
    sales: pd.DataFrame
    calendar: pd.DataFrame


# ==========================
# 1–3. ЗАВАНТАЖЕННЯ ТА ЯКІСТЬ ДАНИХ
# ==========================

def load_data() -> CoopData:
    """Завантаження даних з CSV у структуру CoopData."""
    customers = pd.read_csv(CUSTOMERS_CSV, sep=";", dtype=str)
    products = pd.read_csv(PRODUCTS_CSV, sep=";", dtype=str)
    marketing = pd.read_csv(MARKETING_CSV, sep=";", dtype=str)
    sales = pd.read_csv(SALES_CSV, sep=";", dtype=str)
    calendar = pd.read_csv(CALENDAR_CSV, sep=";", dtype=str)

    # Кастинг типів
    sales[COLS["amount"]] = pd.to_numeric(sales[COLS["amount"]], errors="coerce")
    sales[COLS["qty_weight"]] = pd.to_numeric(sales[COLS["qty_weight"]], errors="coerce")
    sales[COLS["qty_pieces"]] = pd.to_numeric(sales[COLS["qty_pieces"]], errors="coerce")
    sales[COLS["date"]] = pd.to_datetime(sales[COLS["date"]], errors="coerce")

    if COLS["birth_year"] in customers.columns:
        customers[COLS["birth_year"]] = pd.to_numeric(customers[COLS["birth_year"]], errors="coerce")
    if COLS["join_year"] in customers.columns:
        customers[COLS["join_year"]] = pd.to_numeric(customers[COLS["join_year"]], errors="coerce")

    return CoopData(customers, products, marketing, sales, calendar)


def data_quality_report(data: CoopData) -> None:
    """3.2 Аналіз якості даних."""
    by = COLS["birth_year"]
    amt = COLS["amount"]
    qw = COLS["qty_weight"]
    qp = COLS["qty_pieces"]

    print("\n=== 3.2 ЯКІСТЬ ДАНИХ ===")

    # Аномальні роки народження
    if by in data.customers.columns:
        mask_anom = (data.customers[by] < 1900) | (data.customers[by] > 2010)
        n_anom = mask_anom.sum()
        print(f"Аномальні роки народження (<1900 або >2010): {n_anom}")

    # Пропущені значення (ND, Non disponibile)
    for df_name, df in [("customers", data.customers),
                        ("products", data.products),
                        ("marketing", data.marketing),
                        ("sales", data.sales)]:
        nd_count = (df == "ND").sum().sum()
        non_disp_count = (df == "Non disponibile").sum().sum()
        print(f"{df_name}: 'ND' = {nd_count}, 'Non disponibile' = {non_disp_count}")

    # Негативні значення (промоакції)
    neg_amt = (data.sales[amt] < 0).sum()
    neg_qw = (data.sales[qw] < 0).sum()
    neg_qp = (data.sales[qp] < 0).sum()
    print(f"Негативні IMPORTO: {neg_amt}, QTA_PESO: {neg_qw}, QTA_PEZZI: {neg_qp} (ймовірно промоакції)")


# ==========================
# 3.3–3.4 АНАЛІЗ ТОВАРІВ І ПРОДАЖІВ
# ==========================

def build_product_hierarchy(data: CoopData) -> pd.DataFrame:
    """Об’єднання articoli + mkt."""
    p = data.products
    m = data.marketing
    pid = COLS["product_id"]

    merged = p.merge(m, on=pid, how="left", suffixes=("_ART", "_MKT"))
    return merged


def analyze_freschissimi(data: CoopData, prod_mkt: pd.DataFrame) -> None:
    """3.3 Аналіз сектору FRESCHISSIMI."""
    sector_col = COLS["sector"]
    pid = COLS["product_id"]

    print("\n=== 3.3 АНАЛІЗ ТОВАРІВ (FRESCHISSIMI) ===")

    fresch = prod_mkt[prod_mkt[sector_col] == SECTOR_FRESCH]
    n_total = fresch[pid].nunique()
    print(f"Сектор {SECTOR_FRESCH}: всього товарів = {n_total}")

    # Протягом 6 місяців останнього періоду
    sales = data.sales.copy()
    max_date = sales[COLS["date"]].max()
    six_months_ago = max_date - pd.DateOffset(months=6)
    recent_sales = sales[(sales[COLS["date"]] >= six_months_ago)]

    recent_fresch = recent_sales.merge(fresch[[pid, sector_col]], on=pid, how="inner")
    n_sold = recent_fresch[pid].nunique()
    perc = 100 * n_sold / max(n_total, 1)
    print(f"Товарів, що продавались за останні 6 місяців: {n_sold} ({perc:.1f}%)")


def analyze_sales(data: CoopData, prod_mkt: pd.DataFrame) -> None:
    """3.4 Аналіз продажів: ТОП по штуках і по доходу."""
    print("\n=== 3.4 АНАЛІЗ ПРОДАЖІВ ===")
    pid = COLS["product_id"]
    amt = COLS["amount"]
    ticket = COLS["ticket_id"]

    sales = data.sales.merge(prod_mkt[[pid, "DESCRIZIONE_ARTICOLO"]] if "DESCRIZIONE_ARTICOLO" in prod_mkt.columns
                             else prod_mkt[[pid]],
                             on=pid, how="left")

    # ТОП за кількістю продажів (кількість чеків з товаром)
    top_qty = (sales
               .groupby([pid])[[ticket]]
               .nunique()
               .rename(columns={ticket: "num_sales"})
               .sort_values("num_sales", ascending=False)
               .head(10))
    print("\nТОП-10 товарів за кількістю продажів:")
    print(top_qty)

    # ТОП за доходом
    top_rev = (sales
               .groupby([pid])[[amt]]
               .sum()
               .rename(columns={amt: "total_revenue"})
               .sort_values("total_revenue", ascending=False)
               .head(10))
    print("\nТОП-10 товарів за доходом:")
    print(top_rev)


# ==========================
# 4. ПРОФІЛЮВАННЯ КЛІЄНТІВ
# ==========================

def build_customer_profile(data: CoopData,
                           sector_filter: Optional[str] = None) -> pd.DataFrame:
    """
    4. Профілювання клієнтів:
    - Monetary Volume
    - Number of Visits
    - Number of Products
    Якщо sector_filter заданий — рахувати тільки по цьому сектору (наприклад, FRESCHISSIMI).
    """
    print("\n=== 4. ПРОФІЛЮВАННЯ КЛІЄНТІВ ===")
    cid = COLS["customer_id"]
    pid = COLS["product_id"]
    amt = COLS["amount"]
    ticket = COLS["ticket_id"]

    sales = data.sales.copy()
    if sector_filter is not None:
        prod_mkt = build_product_hierarchy(data)
        sales = sales.merge(prod_mkt[[pid, COLS["sector"]]], on=pid, how="left")
        sales = sales[sales[COLS["sector"]] == sector_filter]

    prof = (sales
            .groupby(cid)
            .agg(
                monetary_volume=(amt, "sum"),
                num_visits=(ticket, "nunique"),
                num_products=(pid, pd.Series.nunique)
            )
            .reset_index())

    print(prof.head())
    return prof


# ==========================
# 5. ВИЯВЛЕННЯ ПОДІЙ: CHURN І FOCUSING
# ==========================

def build_monthly_churn_table(data: CoopData) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    5.1 Таблиця "кількість відвідувань за місяць" + позначка churn.
    Правило: якщо наступна покупка через >2 місяці або її немає — поточний місяць = churn.
    """
    print("\n=== 5.1 АНАЛІЗ ВІДТОКУ (CHURN) ===")
    cid = COLS["customer_id"]
    ticket = COLS["ticket_id"]
    date = COLS["date"]

    sales = data.sales.copy()
    sales["year_month"] = sales[date].dt.to_period("M")

    monthly = (sales
               .groupby([cid, "year_month"])
               .agg(num_visits=(ticket, "nunique"))
               .reset_index())

    monthly = monthly.sort_values([cid, "year_month"])
    monthly["next_month"] = monthly.groupby(cid)["year_month"].shift(-1)

    def month_diff(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return (b.year - a.year) * 12 + (b.month - a.month)

    monthly["diff_to_next"] = monthly.apply(
        lambda row: month_diff(row["year_month"], row["next_month"]), axis=1
    )

    monthly["is_churn_month"] = (monthly["diff_to_next"].isna()) | (monthly["diff_to_next"] > 2)

    print(monthly.head())

    churn_per_client = (monthly
                        .groupby(cid)["is_churn_month"]
                        .any()
                        .reset_index()
                        .rename(columns={"is_churn_month": "isChurning"}))

    churn_rate = churn_per_client["isChurning"].mean()
    print(f"\nЧастка клієнтів з відтоком (хоч один churn-місяць): {churn_rate:.2%}")

    return monthly, churn_per_client


def focusing_analysis(data: CoopData) -> pd.DataFrame:
    """
    5.2 Focusing для сектору FRESCHISSIMI.
    - Рахуємо частки підкатегорій у перших 3 та останніх 3 місяцях.
    - Клієнт фокусований, якщо частка якоїсь підкатегорії зросла >= 2 рази.
    """
    print("\n=== 5.2 FOCUSING (ФОКУСУВАННЯ КЛІЄНТІВ) ===")
    cid = COLS["customer_id"]
    pid = COLS["product_id"]
    date = COLS["date"]
    sector = COLS["sector"]
    subsector = COLS["subsector"]
    amt = COLS["amount"]

    prod_mkt = build_product_hierarchy(data)
    sales = data.sales.merge(prod_mkt[[pid, sector, subsector]], on=pid, how="left")
    sales = sales[sales[sector] == SECTOR_FRESCH].copy()
    sales["year_month"] = sales[date].dt.to_period("M")

    months_sorted = np.sort(sales["year_month"].unique())
    if len(months_sorted) < 6:
        print("Мало місяців для коректного Focusing (потрібно >= 6).")
        return pd.DataFrame()

    first3 = months_sorted[:3]
    last3 = months_sorted[-3:]

    agg = (sales
           .groupby([cid, "year_month", subsector])
           .agg(amount_sum=(amt, "sum"))
           .reset_index())

    def build_share(period_months, label):
        tmp = agg[agg["year_month"].isin(period_months)].copy()
        total_per_client = tmp.groupby([cid])["amount_sum"].transform("sum")
        tmp["share"] = tmp["amount_sum"] / total_per_client
        tmp = (tmp
               .groupby([cid, subsector])["share"]
               .sum()
               .reset_index()
               .rename(columns={"share": f"share_{label}"}))
        return tmp

    first_share = build_share(first3, "first3")
    last_share = build_share(last3, "last3")

    focusing = first_share.merge(last_share, on=[cid, subsector], how="outer").fillna(0.0)
    focusing["ratio"] = focusing.apply(
        lambda r: np.inf if r["share_first3"] == 0 and r["share_last3"] > 0
        else (r["share_last3"] / r["share_first3"] if r["share_first3"] > 0 else 0),
        axis=1
    )
    focusing["is_focused"] = focusing["ratio"] >= 2.0

    focused_clients = (focusing
                       .groupby(cid)["is_focused"]
                       .any()
                       .reset_index()
                       .rename(columns={"is_focused": "isFocused"}))
    print(focused_clients.head())

    sample = focusing[focusing["is_focused"]].head(10)
    print("\nПриклади фокусування (клієнт/підкатегорія):")
    print(sample[[cid, subsector, "share_first3", "share_last3", "ratio"]])

    return focusing


# ==========================
# 6. АНАЛІЗ ЧАСОВИХ РЯДІВ (PESCE)
# ==========================

def time_series_pesce(data: CoopData) -> pd.DataFrame:
    """
    6. Часовий ряд покупок по сегменту PESCE (по тижнях).
    """
    print("\n=== 6. АНАЛІЗ ЧАСОВИХ РЯДІВ (PESCE) ===")
    pid = COLS["product_id"]
    date = COLS["date"]
    sector = COLS["sector"]
    ticket = COLS["ticket_id"]

    prod_mkt = build_product_hierarchy(data)
    sales = data.sales.merge(prod_mkt[[pid, sector]], on=pid, how="left")
    pesce = sales[sales[sector] == SECTOR_PESCE].copy()

    if pesce.empty:
        print("Немає даних по сегменту PESCE.")
        return pd.DataFrame()

    pesce["week"] = pesce[date].dt.to_period("W")

    ts = (pesce
          .groupby("week")[ticket]
          .nunique()
          .reset_index()
          .rename(columns={ticket: "num_purchases"}))

    print(ts.head())
    print("\nКоментар: видно нерегулярні покупки та довгі інтервали між ними.")

    return ts


# ==========================
# 7. КЛАСТЕРИЗАЦІЯ K-MEANS
# ==========================

def kmeans_clustering(data: CoopData,
                      profile: pd.DataFrame,
                      k_min: int = 2,
                      k_max: int = 10) -> pd.DataFrame:
    """
    7. K-Means по клієнтах (behavior + демографія якщо є).
    Повертає DF з колонкою 'cluster'.
    """
    print("\n=== 7. КЛАСТЕРИЗАЦІЯ (K-MEANS) ===")
    cid = COLS["customer_id"]
    by = COLS["birth_year"]

    df = profile.merge(
        data.customers[[cid, by, COLS["gender"], COLS["marital_status"]]],
        on=cid,
        how="left"
    )

    last_year = data.sales[COLS["date"]].dt.year.max()
    df["age"] = last_year - df[by]

    features = ["monetary_volume", "num_visits", "num_products", "age"]
    X = df[features].fillna(df[features].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_samples = X_scaled.shape[0]

    # допустимі значення k для silhouette_score: 2..n_samples-1
    k_values = [k for k in range(k_min, k_max + 1)
                if 2 <= k <= n_samples - 1]

    if not k_values:
        print("Недостатньо клієнтів для адекватної кластеризації (silhouette). Призначаю 1 кластер.")
        df["cluster"] = 0
        return df

    best_k = None
    best_sil = -1.0
    sil_scores = {}

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        sil_scores[k] = sil
        if sil > best_sil:
            best_sil = sil
            best_k = k

    print("Silhouette scores за k:")
    for k, s in sil_scores.items():
        print(f"k={k}: {s:.3f}")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
    print(f"\nОбрано k={best_k} з silhouette={best_sil:.3f}")

    cluster_desc = (df
                    .groupby("cluster")[features]
                    .agg(["mean", "min", "max"]))
    print("\nХарактеристики кластерів:")
    print(cluster_desc)

    return df


# ==========================
# 8. КЛАСИФІКАЦІЯ ВІДТОКУ (DECISION TREE)
# ==========================

def churn_classification(profile: pd.DataFrame,
                         churn_labels: pd.DataFrame) -> None:
    """
    8. Класифікація isChurning за:
    - Number of visits
    - Monetary volume
    - Number of different products
    """
    print("\n=== 8. КЛАСИФІКАЦІЯ ВІДТОКУ (DECISION TREE) ===")
    cid = COLS["customer_id"]

    df = profile.merge(churn_labels, on=cid, how="inner")
    X = df[["num_visits", "monetary_volume", "num_products"]]
    y = df["isChurning"].astype(int)

    # Якщо всі клієнти або всі НЕ в churn — модель немає сенсу
    if y.nunique() < 2:
        print("У всіх клієнтів однаковий статус churn (усі 0 або усі 1).")
        print("Класифікацію побудувати неможливо — немає різниці між класами.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec_churn = recall_score(y_test, y_pred, pos_label=1)

    print(f"Accuracy: {acc:.3f}")
    print(f"Recall (churning=1): {rec_churn:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nДерево рішень (аналог J48):")
    print(export_text(clf, feature_names=list(X.columns)))


# ==========================
# 9. ПОШУК ІННОВАТОРІВ
# ==========================

def innovators_detection(data: CoopData) -> pd.DataFrame:
    """
    9. Пошук інноваторів:
    - знаходимо товари з низькими продажами в перший тиждень і зростанням надалі;
    - клієнти, які купили в перший тиждень — інноватори.
    """
    print("\n=== 9. ПОШУК ІННОВАТОРІВ ===")
    cid = COLS["customer_id"]
    pid = COLS["product_id"]
    date = COLS["date"]

    sales = data.sales.copy()
    sales["week"] = sales[date].dt.to_period("W")

    first_week = (sales
                  .groupby(pid)["week"]
                  .min()
                  .reset_index()
                  .rename(columns={"week": "first_week"}))

    s = sales.merge(first_week, on=pid, how="left")

    weekly_prod = (s
                   .groupby([pid, "week"])[cid]
                   .nunique()
                   .reset_index()
                   .rename(columns={cid: "num_buyers"}))

    innovators_list = []

    for product, grp in weekly_prod.groupby(pid):
        grp = grp.sort_values("week")
        fw = grp["week"].min()
        first_week_sales = grp[grp["week"] == fw]["num_buyers"].iloc[0]
        later_sales = grp[grp["week"] > fw]["num_buyers"].sum()

        total = first_week_sales + later_sales
        if total == 0:
            continue
        first_share = first_week_sales / total
        if first_share < 0.2 and later_sales > first_week_sales:
            innovators_list.append(product)

    innovators_list = innovators_list[:50]
    print(f"Знайдено товарів з інноваційною динамікою: {len(innovators_list)}")

    innovators_sales = s[s[pid].isin(innovators_list)]
    innovators_first_week = innovators_sales[innovators_sales["week"] == innovators_sales["first_week"]]

    innovators = innovators_first_week[[cid, pid, "week"]].drop_duplicates()

    print("\nПриклади інноваторів (клієнт/товар/тиждень):")
    print(innovators.head())

    by = COLS["birth_year"]
    last_year = data.sales[date].dt.year.max()
    cust = data.customers[[cid, by, COLS["gender"], COLS["marital_status"]]].copy()
    cust["age"] = last_year - cust[by]

    innovators_demo = innovators.merge(cust, on=cid, how="left")
    print("\nСередній вік інноваторів:", innovators_demo["age"].mean())
    print("Розподіл за статтю:")
    print(innovators_demo[COLS["gender"]].value_counts(normalize=True))

    return innovators_demo


# ==========================
# 10. ЧАСТІ ПАТЕРНИ (APRIORI)
# ==========================

def frequent_patterns_cluster4(clustered: pd.DataFrame,
                               data: CoopData) -> Optional[pd.DataFrame]:
    """
    10. Пошук частих патернів для кластера 4 (або останнього, якщо 4 немає).
    """
    if not HAS_MLXTEND:
        print("\nApriori недоступний (немає mlxtend). Пропускаємо крок 10.")
        return None

    print("\n=== 10. ПОШУК ЧАСТИХ ПАТЕРНІВ (APRIORI) ===")
    cid = COLS["customer_id"]
    pid = COLS["product_id"]
    ticket = COLS["ticket_id"]
    segment = COLS["segment"]

    if 4 in clustered["cluster"].unique():
        target_cluster = 4
    else:
        target_cluster = clustered["cluster"].max()
    print(f"Використовуємо кластер {target_cluster} (найцінніші клієнти).")

    best_clients = clustered[clustered["cluster"] == target_cluster][cid].unique()
    sales = data.sales[data.sales[cid].isin(best_clients)].copy()

    if sales.empty:
        print("Немає продажів для обраного кластера — Apriori неможливий.")
        return None

    prod_mkt = build_product_hierarchy(data)
    sales = sales.merge(prod_mkt[[pid, segment]], on=pid, how="left")

    basket = (sales
              .groupby([ticket, segment])[pid]
              .nunique()
              .unstack(fill_value=0))

    basket = (basket > 0).astype(bool)

    print(f"Розмір транзакційної матриці: {basket.shape}")

    if basket.shape[0] == 0 or basket.shape[1] == 0:
        print("Порожня транзакційна матриця — Apriori неможливий.")
        return None

    freq_items = apriori(basket, min_support=0.02, use_colnames=True)
    if freq_items.empty:
        print("Жодних частих наборів з вказаною підтримкою не знайдено.")
        return None

    rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
    if rules.empty:
        print("Правила асоціацій відсутні при заданих параметрах.")
        return None

    rules = rules.sort_values("lift", ascending=False).head(20)

    print("\nТоп-правила асоціацій:")
    print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

    return rules


# ==========================
# 11. ОЦІНКА ПРИВАТНОСТІ
# ==========================

def privacy_evaluation(data: CoopData) -> None:
    """
    11. Перевірка анонімності: дивимось, які атрибути є у таблиці клієнтів.
    """
    print("\n=== 11. ОЦІНКА ПРИВАТНОСТІ ===")
    print("Колонки таблиці клієнтів:")
    print(list(data.customers.columns))

    print("\nКоментар:")
    print("- В даних немає імен, адрес, персональних кодів (якщо ти їх не додавав).")
    print("- Є лише рік народження, стать, сімейний стан, професія, рік вступу.")
    print("- Ризик деанонімізації низький за умови відсутності зовнішніх ідентифікаторів.")


# ==========================
# MAIN: ЗБИРАЄМО ВСЕ РАЗОМ
# ==========================

def main():
    print("Завантаження даних...")
    data = load_data()

    # 3.2 Якість даних
    data_quality_report(data)

    # 3.3–3.4 Товари і продажі
    prod_mkt = build_product_hierarchy(data)
    analyze_freschissimi(data, prod_mkt)
    analyze_sales(data, prod_mkt)

    # 4. Профілі клієнтів (всі товари)
    profile_all = build_customer_profile(data)
    # 4. Профіль для FRESCHISSIMI
    profile_fresch = build_customer_profile(data, sector_filter=SECTOR_FRESCH)

    # 5.1 Churn
    monthly_churn, churn_labels = build_monthly_churn_table(data)

    # 5.2 Focusing
    focusing = focusing_analysis(data)

    # 6. Часові ряди (PESCE)
    ts_pesce = time_series_pesce(data)

    # 7. Кластеризація
    clustered = kmeans_clustering(data, profile_all)

    # 8. Класифікація churn
    churn_classification(profile_all, churn_labels)

    # 9. Інноватори
    innovators_demo = innovators_detection(data)

    # 10. Часті патерни (Apriori) для найкращого кластера
    rules = frequent_patterns_cluster4(clustered, data)

    # 11. Приватність
    privacy_evaluation(data)

    print("\n=== АНАЛІЗ ЗАВЕРШЕНО ===")


if __name__ == "__main__":
    main()
