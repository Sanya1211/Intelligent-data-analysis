import psycopg2
import pandas as pd
from datetime import date
import warnings

# Приглушити конкретний warning від pandas про SQLAlchemy
warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable",
    category=UserWarning,
)
# ===== Налаштування підключення до PostgreSQL =====
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "ads_db",
    "user": "postgres",
    "password": "1111",  # пароль МАЄ бути РЯДКОМ, заміни на свій
}


# ===== Сервісні функції =====
def get_connection():
    """Повертає підключення до БД або кидає виняток."""
    return psycopg2.connect(**DB_CONFIG)


def init_db() -> bool:
    """Створює таблицю, якщо її ще немає. Повертає True/False залежно від успіху."""
    ddl = """
    CREATE TABLE IF NOT EXISTS project_materials (
        id              SERIAL PRIMARY KEY,
        project_name    TEXT        NOT NULL,
        client_name     TEXT        NOT NULL,
        address         TEXT        NOT NULL,
        start_date      DATE        NOT NULL,
        material_name   TEXT        NOT NULL,
        unit            TEXT        NOT NULL,
        unit_price_usd  NUMERIC(12,2) NOT NULL,
        exchange_rate   NUMERIC(12,4) NOT NULL,
        quantity        NUMERIC(12,2) NOT NULL
    );
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
        print("✅ Таблиця project_materials готова.")
        return True

    except psycopg2.OperationalError as e:
        print("❌ Не вдалося підключитися до бази даних.")
        print("   Перевірте, що PostgreSQL запущений і налаштування DB_CONFIG коректні.")
        print("   Технічні деталі:", e)
        return False

    except Exception as e:
        print("❌ Сталася помилка при створенні таблиці project_materials.")
        print("   Технічні деталі:", e)
        return False


def insert_project_material(
    project_name: str,
    client_name: str,
    address: str,
    start_date: date,
    material_name: str,
    unit: str,
    unit_price_usd: float,
    exchange_rate: float,
    quantity: float,
):
    """Додає один запис про використання матеріалу в проекті."""
    sql = """
    INSERT INTO project_materials (
        project_name, client_name, address, start_date,
        material_name, unit, unit_price_usd, exchange_rate, quantity
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        project_name,
                        client_name,
                        address,
                        start_date,
                        material_name,
                        unit,
                        unit_price_usd,
                        exchange_rate,
                        quantity,
                    ),
                )
            conn.commit()

    except psycopg2.OperationalError as e:
        print("❌ Не вдалося підключитися до бази даних під час вставки запису.")
        print("   Запис не збережено. Технічні деталі:", e)

    except Exception as e:
        print("❌ Помилка при вставці запису в project_materials.")
        print("   Перевірте коректність даних. Технічні деталі:", e)


def get_year_dataframe(year: int) -> pd.DataFrame:
    """
    Повертає DataFrame з усіма розрахунковими полями для заданого року.
    Якщо сталася помилка або даних немає, повертає порожній DataFrame.
    """
    query = """
    SELECT
        id,
        project_name,
        client_name,
        address,
        start_date,
        EXTRACT(QUARTER FROM start_date)::int AS quarter,
        material_name,
        unit,
        unit_price_usd,
        exchange_rate,
        quantity,
        (unit_price_usd * exchange_rate)                 AS unit_price_uah,
        (quantity * unit_price_usd)                      AS total_usd,
        (quantity * unit_price_usd * exchange_rate)      AS total_uah
    FROM project_materials
    WHERE EXTRACT(YEAR FROM start_date) = %s
    ORDER BY start_date, project_name, material_name;
    """
    try:
        with get_connection() as conn:
            df = pd.read_sql(query, conn, params=(year,))
        return df

    except psycopg2.OperationalError as e:
        print("❌ Не вдалося підключитися до бази даних при читанні даних за рік.")
        print("   Технічні деталі:", e)

    except Exception as e:
        print("❌ Сталася помилка при читанні даних з бази.")
        print("   Технічні деталі:", e)

    print(f"ℹ️ Дані за {year} рік недоступні через помилку.")
    return pd.DataFrame()


def build_pivot(df: pd.DataFrame, quarter: int) -> pd.DataFrame:
    """
    Будує зведену таблицю:
      - фільтр за кварталом
      - index = матеріал
      - columns = клієнти
      - values = сума total_uah
    """
    if df.empty:
        print("ℹ️ Немає даних для побудови зведеної таблиці. Спочатку додайте записи до БД.")
        return pd.DataFrame()

    df_q = df[df["quarter"] == quarter]

    if df_q.empty:
        print(f"ℹ️ За {quarter} квартал у вибраному році записів немає.")
        return pd.DataFrame()

    pivot = df_q.pivot_table(
        index="material_name",
        columns="client_name",
        values="total_uah",
        aggfunc="sum",
        fill_value=0,
    )

    pivot.index.name = "Матеріал"
    pivot.columns.name = "Клієнт"

    return pivot


# ===== Допоміжна функція для безпечного вводу чисел =====
def safe_input_int(prompt: str, min_value=None, max_value=None) -> int:
    """
    Просить користувача ввести ціле число.
    Якщо введення некоректне - повторює запит з підказкою.
    """
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("⚠️ Потрібно ввести ціле число. Спробуйте ще раз.")
            continue

        if min_value is not None and value < min_value:
            print(f"⚠️ Число не може бути меншим за {min_value}. Спробуйте ще раз.")
            continue
        if max_value is not None and value > max_value:
            print(f"⚠️ Число не може бути більшим за {max_value}. Спробуйте ще раз.")
            continue

        return value


# ===== Демонстраційне заповнення даними за кілька років =====
def demo_fill_data():
    """
    Демонстраційне заповнення БД даними за КІЛЬКА РОКІВ.
    Для простоти:
      - очищаємо таблицю;
      - додаємо ті самі 2 проекти для трьох років: (поточний-2), (поточний-1), (поточний).
    """
    today = date.today()
    exchange_rate = 40.0  # умовний курс
    years = [today.year - 2, today.year - 1, today.year]

    try:
        # Очищаємо таблицю, щоб демо-дані не дублювалися
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM project_materials;")
            conn.commit()

        for y in years:
            # 1-й проект
            insert_project_material(
                project_name=f"Branding ТРЦ Ocean ({y})",
                client_name="ТОВ 'Маркетингові рішення'",
                address="м. Київ, пр. Перемоги, 10",
                start_date=date(y, 2, 15),
                material_name="Банер 3x6",
                unit="шт",
                unit_price_usd=50.0,
                exchange_rate=exchange_rate,
                quantity=10,
            )

            # 2-й проект
            insert_project_material(
                project_name=f"Лайтбокси для магазину ({y})",
                client_name="ТОВ 'Магазин одягу'",
                address="м. Київ, вул. Хрещатик, 20",
                start_date=date(y, 5, 10),
                material_name="Лайтбокс 1x2",
                unit="шт",
                unit_price_usd=120.0,
                exchange_rate=exchange_rate,
                quantity=5,
            )

        print(f"✅ Демо-дані додано за роки: {', '.join(map(str, years))}.")

    except Exception as e:
        print("❌ Сталася помилка під час додавання демо-даних.")
        print("   Технічні деталі:", e)


def main():
    if not init_db():
        print("Програма не може працювати без таблиці в базі даних. Завершення.")
        return

    while True:
        print("\n=== Меню ===")
        print("1 - Заповнити демо-даними (за кілька років)")
        print("2 - Показати всі дані за рік")
        print("3 - Показати зведену таблицю (квартал/матеріал/клієнт)")
        print("0 - Вихід")
        choice = input("Ваш вибір: ").strip()

        if choice == "0":
            print("До побачення!")
            break

        elif choice == "1":
            demo_fill_data()

        elif choice == "2":
            year = safe_input_int("Вкажіть рік (наприклад 2023): ", 1900, 2100)
            df = get_year_dataframe(year)
            if df.empty:
                print(f"ℹ️ За {year} рік даних немає або сталася помилка при читанні.")
            else:
                print("\nПовна таблиця:")
                print(
                    df[
                        [
                            "project_name",
                            "client_name",
                            "address",
                            "start_date",
                            "quarter",
                            "material_name",
                            "unit",
                            "unit_price_usd",
                            "unit_price_uah",
                            "quantity",
                            "total_usd",
                            "total_uah",
                        ]
                    ]
                )

        elif choice == "3":
            year = safe_input_int("Рік: ", 1900, 2100)
            quarter = safe_input_int("Квартал (1-4): ", 1, 4)
            df = get_year_dataframe(year)
            if df.empty:
                print(f"ℹ️ За {year} рік даних немає або сталася помилка при читанні.")
                continue

            pivot = build_pivot(df, quarter)
            if not pivot.empty:
                print(f"\nЗведена таблиця за {quarter} квартал {year} року (сума в грн):")
                print(pivot)

        else:
            print("⚠️ Невірний вибір, спробуйте ще.")


if __name__ == "__main__":
    main()
