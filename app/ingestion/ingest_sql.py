import os
import sqlite3
import pandas as pd
from typing import Literal


# 📍 Get base directory (this file's location)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# 📁 Set path for database and default CSV
DB_FOLDER = os.path.abspath(os.path.join(BASE_DIR, "../../database"))
DB_FILE_PATH = os.path.join(DB_FOLDER, "supplychain.db")
DEFAULT_CSV = os.path.abspath(os.path.join(BASE_DIR, "../../data/demand_data.csv"))
TABLE_NAME = "demand"

# 📦 Ensure the database folder exists
os.makedirs(DB_FOLDER, exist_ok=True)


def ingest_dataframe_to_sqlite(
    df: pd.DataFrame,
    mode: Literal["append", "overwrite"] = "append"
) -> None:
    """
    Write a DataFrame to SQLite.
    
    Parameters:
    - df: DataFrame to insert
    - mode: "append" to add rows, "overwrite" to drop and replace the table
    """
    try:
        # 🔌 Connect to SQLite
        conn = sqlite3.connect(DB_FILE_PATH)
        cursor = conn.cursor()
        print(f"✅ Connected to SQLite at {DB_FILE_PATH}")

        # 🧹 Drop table if overwrite is selected
        if mode == "overwrite":
            print(f"⚠️ Overwriting '{TABLE_NAME}' table...")
            cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")

        # 💾 Insert DataFrame
        df.to_sql(TABLE_NAME, conn, if_exists="append" if mode == "append" else "replace", index=False)
        conn.commit()
        conn.close()

        print(f"✅ Table '{TABLE_NAME}' written successfully in {mode} mode.")

    except Exception as e:
        print(f"❌ Failed to ingest DataFrame: {e}")


def ingest_csv_to_sqlite_from_path(
    csv_path: str,
    mode: Literal["append", "overwrite"] = "append"
) -> None:
    """
    Load CSV file and ingest into SQLite DB.
    
    Parameters:
    - csv_path: Path to your CSV file
    - mode: Ingestion mode ("append" or "overwrite")
    """
    try:
        print(f"📥 Loading CSV from: {csv_path}")
        # Try multiple encodings to handle non-UTF8 files
        for enc in ("utf-8", "utf-8-sig", "latin1"):
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                if enc != "utf-8":
                    print(f"⚠️ Loaded CSV with '{enc}' encoding due to decode issues")
                break
            except UnicodeDecodeError:
                continue
        else:
            print("❌ Failed to load CSV in any of the tried encodings: utf-8, utf-8-sig, latin1")
            return
        ingest_dataframe_to_sqlite(df, mode=mode)
    except Exception as e:
        print(f"❌ Failed to read and ingest CSV: {e}")


# 🏃‍♂️ Execute if run directly
if __name__ == "__main__":
    ingest_csv_to_sqlite_from_path(DEFAULT_CSV, mode="overwrite")
