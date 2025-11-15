import pandas as pd
import matplotlib.pyplot as plt


def build_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Обчислення показників (measures) та вимірів часу."""

    # Показники
    df["total_cost_usd"] = df["capex_usd"] + df["opex_usd"]
    df["profit_usd_year"] = df["revenue_usd_year"] - df["opex_usd"]

    # Захист від ділення на 0 / від'ємного прибутку
    df["profit_usd_year_clipped"] = df["profit_usd_year"].clip(lower=1)
    df["payback_years"] = df["capex_usd"] / df["profit_usd_year_clipped"]
    df["lcoe_usd_per_kwh"] = df["total_cost_usd"] / df["energy_kwh_year"]

    # Вимір Time
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)

    return df


def build_and_print_olap(df: pd.DataFrame):
    """
    Формує OLAP-таблиці (куби), друкує їх у консоль
    і повертає як об'єкти pandas.
    """

    print("=== Перші рядки факт-таблиці ===")
    print(df.head(), "\n")

    # ----- Куб Product x Time -----
    cube_product_time = pd.pivot_table(
        df,
        index=["product_category", "product_subcategory"],  # вимір Product
        columns=["year"],                                   # вимір Time
        values=[
            "energy_kwh_year",
            "total_cost_usd",
            "profit_usd_year",
            "payback_years",
            "lcoe_usd_per_kwh",
        ],
        aggfunc={
            "energy_kwh_year": "sum",
            "total_cost_usd": "sum",
            "profit_usd_year": "sum",
            "payback_years": "mean",
            "lcoe_usd_per_kwh": "mean",
        },
    )

    print("=== Куб Product x Time ===")
    print(cube_product_time, "\n")

    # ----- Варіант 1: категорія + підкатегорія -----
    VARIANT_CATEGORY = "Residential"
    VARIANT_SUBCATEGORY = "Roof-mono"

    slice_product = df[
        (df["product_category"] == VARIANT_CATEGORY)
        & (df["product_subcategory"] == VARIANT_SUBCATEGORY)
    ]

    variant1_cube = pd.pivot_table(
        slice_product,
        index=["year"],        # Time
        columns=["country"],   # Region (країна)
        values=[
            "energy_kwh_year",
            "total_cost_usd",
            "profit_usd_year",
            "payback_years",
            "lcoe_usd_per_kwh",
        ],
        aggfunc="mean",
    )

    print(
        f"=== Варіант 1: {VARIANT_CATEGORY} -> {VARIANT_SUBCATEGORY} (year x country) ==="
    )
    print(variant1_cube, "\n")

    # ----- Drill Up -----
    drill_up = (
        df.groupby(["product_category", "year"])
        .agg(
            energy_kwh_year_sum=("energy_kwh_year", "sum"),
            total_cost_usd_sum=("total_cost_usd", "sum"),
            profit_usd_year_sum=("profit_usd_year", "sum"),
        )
        .reset_index()
    )

    print("=== Drill Up: категорія продукту x рік ===")
    print(drill_up.head(), "\n")

    # ----- Drill Down -----
    drill_down = (
        df.groupby(["product_category", "product_subcategory", "city", "year"])
        .agg(
            energy_kwh_year_sum=("energy_kwh_year", "sum"),
            total_cost_usd_sum=("total_cost_usd", "sum"),
            profit_usd_year_sum=("profit_usd_year", "sum"),
        )
        .reset_index()
    )

    print("=== Drill Down: категорія -> підкатегорія -> місто x рік ===")
    print(drill_down.head(), "\n")

    return cube_product_time, variant1_cube, drill_up, drill_down


def save_to_excel(
    df: pd.DataFrame,
    cube_product_time: pd.DataFrame,
    variant1_cube: pd.DataFrame,
    drill_up: pd.DataFrame,
    drill_down: pd.DataFrame,
    filename: str = "solar_olap_results.xlsx",
) -> None:
    """
    Зберігає факт-таблицю та всі OLAP-таблиці в один Excel-файл
    на різні аркуші.
    """

    # engine="openpyxl" – щоб точно працювало з .xlsx
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Факт-таблиця
        df.to_excel(writer, sheet_name="fact_table", index=False)

        # Куби та агрегації
        cube_product_time.to_excel(writer, sheet_name="cube_product_time")
        variant1_cube.to_excel(writer, sheet_name="variant1_variant")
        drill_up.to_excel(writer, sheet_name="drill_up", index=False)
        drill_down.to_excel(writer, sheet_name="drill_down", index=False)

    print(f"Excel-файл зі звітом збережено як: {filename}\n")


def plot_visualizations(df: pd.DataFrame) -> None:
    """Декілька графіків для звіту."""

    # ---- 1. Сумарний прибуток за підкатегоріями ----
    profit_by_subcat = (
        df.groupby("product_subcategory")["profit_usd_year"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    profit_by_subcat.plot(kind="bar")
    plt.title("Сумарний річний прибуток за підкатегоріями панелей")
    plt.ylabel("Прибуток, $/рік")
    plt.xlabel("Підкатегорія")
    plt.tight_layout()
    plt.show()

    # ---- 2. Середній строк окупності по роках і категоріях ----
    payback_by_year_cat = (
        df.groupby(["year", "product_category"])["payback_years"]
        .mean()
        .unstack("product_category")
    )

    plt.figure(figsize=(8, 5))
    for col in payback_by_year_cat.columns:
        plt.plot(
            payback_by_year_cat.index,
            payback_by_year_cat[col],
            marker="o",
            label=col,
        )
    plt.title("Середній строк окупності за роками")
    plt.ylabel("Роки")
    plt.xlabel("Рік")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- 3. Виробіток енергії для варіанту 1 (Residential, Roof-mono) ----
    cat, subcat = "Residential", "Roof-mono"
    mask = (df["product_category"] == cat) & (df["product_subcategory"] == subcat)
    energy_by_year = df[mask].groupby("year")["energy_kwh_year"].sum()

    plt.figure(figsize=(8, 5))
    energy_by_year.plot(kind="bar")
    plt.title(f"Річний виробіток для {cat} - {subcat}")
    plt.ylabel("кВт·год/рік")
    plt.xlabel("Рік")
    plt.tight_layout()
    plt.show()


def main():
    file_name = "solar_panels.csv"  # CSV має лежати в цій самій папці

    # 1. Зчитуємо дані
    df = pd.read_csv(file_name, parse_dates=["date"])

    # 2. Обчислюємо показники та вимір часу
    df = build_measures(df)

    # 3. Формуємо та друкуємо OLAP-таблиці
    cube_product_time, variant1_cube, drill_up, drill_down = build_and_print_olap(df)

    # 4. Зберігаємо все в Excel
    save_to_excel(df, cube_product_time, variant1_cube, drill_up, drill_down)

    # 5. Візуалізації
    plot_visualizations(df)


if __name__ == "__main__":
    main()
