"""Main Streamlit application for the Supply Chain Assistant."""
import warnings
# Suppress LangChain deprecation warnings for memory and Chain.run
warnings.filterwarnings("ignore", category=DeprecationWarning)
import streamlit as st
from query_router import handle_user_query
from ingestion.ingest_sql import ingest_csv_to_sqlite_from_path
from ingestion.ingest_unstructured import embed_and_store_unstructured_files, PINECONE_INDEX_NAME, NAMESPACE
from pinecone import Pinecone
import sqlite3
import pandas as pd
import numpy as np
# Try Prophet import; fallback if unavailable
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
from scipy.stats import norm
from datetime import datetime
import os
import json
import openai
import re
from dotenv import load_dotenv

# Load environment variables and set OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define path to the supply chain database
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DB_PATH = os.path.join(BASE_DIR, "database", "supplychain.db")
# Define paths for data files and upload directories
DATA_DIR = os.path.join(BASE_DIR, "data")
DEMAND_CSV_PATH = os.path.join(DATA_DIR, "demand_data.csv")
UPLOADED_DOCS_DIR = os.path.join(DATA_DIR, "uploaded_docs")
# Ensure data directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADED_DOCS_DIR, exist_ok=True)

# 1️⃣ Page config (must be first Streamlit call)
st.set_page_config(page_title="📦 Supply Chain Assistant", layout="wide")

# 


# Sidebar navigation
st.sidebar.title("Navigate")
 # Sidebar navigation (label must be non-empty to avoid warning, hidden via label_visibility)
page = st.sidebar.radio(
     label=" ",  # non-empty space, visibility collapsed
     options=[
        "Chat",
        "Data Upload",
        "Forecasting",
        "Inventory Optimization",
        "Alerts",
        "Dashboard",
        "PO Management",
    ],
    label_visibility="collapsed",
)

if page == "Chat":
    st.title("📦 Predictive Supply Chain Assistant")
    st.markdown(
        """
    Ask questions like:
    - **Forecast quantity for SKU01**
    - **What SKUs need reordering?**
    - **Summarize the uploaded contract**
    """
    )
    # Uploaded Files dropdown with download & delete actions
    with st.expander("📂 Uploaded Files", expanded=False):
        # Gather structured and unstructured files
        try:
            structured = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
        except Exception:
            structured = []
        try:
            unstructured = os.listdir(UPLOADED_DOCS_DIR)
        except Exception:
            unstructured = []
        files = [("Structured", f) for f in structured] + [("Unstructured", f) for f in unstructured]
        if files:
            choices = [f"{kind}: {fname}" for kind, fname in files]
            selection = st.selectbox("Select file", choices, key="chat_file_select")
            kind, fname = selection.split(": ", 1)
            dir_path = DATA_DIR if kind == "Structured" else UPLOADED_DOCS_DIR
            fpath = os.path.join(dir_path, fname)
            col_download, col_delete = st.columns(2)
            with col_download:
                try:
                    with open(fpath, "rb") as f:
                        data_bytes = f.read()
                    mime = "text/csv" if kind == "Structured" else (
                        "application/pdf" if fname.lower().endswith(".pdf") else "text/plain"
                    )
                    st.download_button(f"Download {fname}", data=data_bytes, file_name=fname, mime=mime)
                except Exception:
                    st.error(f"Unable to load {fname}")
            with col_delete:
                if st.button(f"Delete {fname}", key=f"delete_{kind}_{fname}"):
                    # Delete file
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        st.error(f"Error deleting file: {e}")
                        st.stop()
                    if kind == "Structured":
                        # Refresh SQLite 'demand' table
                        remaining = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
                        conn = sqlite3.connect(DB_PATH)
                        cur = conn.cursor()
                        cur.execute("DROP TABLE IF EXISTS demand")
                        conn.commit()
                        conn.close()
                        for idx, rf in enumerate(remaining):
                            mode = "overwrite" if idx == 0 else "append"
                            ingest_csv_to_sqlite_from_path(os.path.join(DATA_DIR, rf), mode=mode)
                        st.success(f"Deleted {fname} and refreshed database from remaining CSVs.")
                    else:
                        # Delete embeddings for unstructured file
                        try:
                            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                            idx = pc.Index(PINECONE_INDEX_NAME)
                            idx.delete(filter={"source": {"$eq": fpath}}, namespace=NAMESPACE)
                            st.success(f"Deleted {fname} and its embeddings")
                        except Exception as e:
                            st.error(f"Error deleting embeddings: {e}")
        else:
            st.info("No uploaded files available.")
    hybrid_mode = st.sidebar.checkbox("Enable hybrid retrieval", value=False)
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    for msg in st.session_state['messages']:
        if msg['role'] == 'user':
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")
    with st.form(key="chat_form", clear_on_submit=True):
        user_query = st.text_input("🔍 Enter your question")
        submitted = st.form_submit_button("Ask")
    if submitted:
        if user_query:
            st.session_state['messages'].append({'role': 'user', 'content': user_query})
            with st.spinner("Thinking..."):
                answer = handle_user_query(user_query, hybrid=hybrid_mode)
            st.session_state['messages'].append({'role': 'assistant', 'content': answer})
            st.markdown(f"**You:** {user_query}")
            st.markdown(f"**Assistant:** {answer}")
        else:
            st.warning("Please enter a question!")

elif page == "Data Upload":
    st.header("📂 Upload Data")
    # Structured data upload: allow multiple CSVs and choose ingestion mode
    structured_files = st.file_uploader(
        "Upload CSV(s) for Structured Data", type="csv", accept_multiple_files=True
    )
    if structured_files:
        ingest_mode = st.radio(
            "Ingestion mode", ("append", "overwrite"), index=0, horizontal=True, key="ingest_mode"
        )
        for idx, uploaded_file in enumerate(structured_files):
            # Determine save path under data directory
            save_name = uploaded_file.name
            save_path = os.path.join(DATA_DIR, save_name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.write(f"✅ Uploaded {save_name}")
            # For overwrite mode, drop table only once (first file), then append subsequent
            mode = "append"
            if ingest_mode == "overwrite" and idx == 0:
                mode = "overwrite"
            # Ingest into SQLite
            ingest_csv_to_sqlite_from_path(save_path, mode=mode)
        st.success("Structured data ingested ✅")
    st.markdown("---")
    unstructured_files = st.file_uploader(
        "Upload PDFs or TXTs", type=["pdf", "txt"], accept_multiple_files=True
    )
    if unstructured_files:
        # Save uploaded docs to project uploaded_docs directory
        uploaded_paths = []
        for file in unstructured_files:
            save_path = os.path.join(UPLOADED_DOCS_DIR, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getvalue())
            uploaded_paths.append(save_path)
        embed_and_store_unstructured_files(uploaded_paths)
        st.success("Unstructured files embedded ✅")

elif page == "Forecasting":
    st.header("📈 Demand Forecasting & Trend Analysis")
    # Data source selection
    data_source = st.radio("Data Source", ("Database Table", "CSV File"), horizontal=True)
    # Load historical data
    if data_source == "Database Table":
        @st.cache_data
        def load_db_history():
            conn = sqlite3.connect(DB_PATH)
            df_db = pd.read_sql_query("SELECT date, quantity FROM demand ORDER BY date", conn)
            conn.close()
            df_db.columns = ["ds", "y"]
            df_db["ds"] = pd.to_datetime(df_db["ds"])
            return df_db
        try:
            df = load_db_history()
        except Exception:
            st.warning(
                "Could not load historical demand data from database.\n"
                "Please upload a CSV with 'date' and 'quantity' columns on the Data Upload page."
            )
            st.stop()
    else:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if not csv_files:
            st.warning("No CSV files found in data directory. Please upload one on the Data Upload page.")
            st.stop()
        selected_csv = st.selectbox("Select CSV file", csv_files)
        csv_path = os.path.join(DATA_DIR, selected_csv)
        # Try CSV with multiple encodings
        for enc in ("utf-8", "utf-8-sig", "latin1"):
            try:
                df_raw = pd.read_csv(csv_path, encoding=enc)
                if enc != "utf-8":
                    st.info(f"Loaded CSV with '{enc}' encoding due to decode issues.")
                break
            except Exception:
                continue
        else:
            st.error("❌ Could not load CSV with encodings: utf-8, utf-8-sig, latin1")
            st.stop()
        cols = df_raw.columns.tolist()
        x_col = st.selectbox("Feature column (X)", cols)
        y_col = st.selectbox("Target column (Y)", [c for c in cols if c != x_col])
        df = df_raw[[x_col, y_col]].copy()
        # Ensure numeric for forecasting
        df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        if df[[x_col, y_col]].isnull().any().any():
            st.error("Selected columns must be numeric for forecasting. Non-numeric values detected.")
            st.stop()
        st.subheader("Historical Data")
        st.line_chart(df.set_index(x_col)[y_col])
        periods = st.slider("Forecast horizon (# future points)", 1, 100, 10)
        # Simple linear regression forecast
        import numpy as np
        slope, intercept = np.polyfit(df[x_col], df[y_col], 1)
        last_x = df[x_col].max()
        future_x = np.arange(last_x + 1, last_x + 1 + periods)
        future_y = slope * future_x + intercept
        forecast_df = pd.DataFrame({x_col: future_x, "yhat": future_y})
        st.subheader("Forecast")
        st.line_chart(forecast_df.set_index(x_col)["yhat"])
        # Stop further processing for CSV-based forecasting
        st.stop()
    # Display historical data
    st.subheader("Historical Data")
    st.line_chart(df.set_index("ds")["y"])
    # Forecast horizon
    periods = st.slider("Forecast horizon (days)", 7, 180, 30)
    if PROPHET_AVAILABLE:
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        st.subheader("Forecasted Demand (Prophet)")
        fig1 = m.plot(forecast)
        st.pyplot(fig1)
        st.subheader("Forecast Components")
        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)
    else:
        st.warning("`prophet` library not installed. Showing simple moving average forecast.")
        window = st.slider("Moving average window (days)", min_value=1, max_value=periods, value=min(7, periods))
        ma = df["y"].rolling(window=window).mean().iloc[-1]
        future_dates = pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=periods)
        forecast_df = pd.DataFrame({"ds": future_dates, "yhat": ma})
        st.subheader("Moving Average Forecast")
        st.line_chart(forecast_df.set_index("ds")["yhat"])

elif page == "Inventory Optimization":
    st.header("🏭 Inventory Optimization")
    # Data source selection
    data_source = st.radio("Data Source", ("Database Table", "CSV File"), horizontal=True, key="inv_data_source")
    # Load demand data
    if data_source == "Database Table":
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("SELECT date, quantity FROM demand ORDER BY date", conn)
            conn.close()
        except Exception:
            st.warning(
                "Could not load demand data from database.\n"
                "Please upload a CSV with 'date' and 'quantity' columns on the Data Upload page."
            )
            st.stop()
    else:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if not csv_files:
            st.warning("No CSV files found in data directory. Please upload one on the Data Upload page.")
            st.stop()
        selected_csv = st.selectbox("Select CSV file", csv_files, key="inv_select_csv")
        csv_path = os.path.join(DATA_DIR, selected_csv)
        try:
            df_raw = pd.read_csv(csv_path)
        except UnicodeDecodeError:
            st.warning("⚠️ UnicodeDecodeError encountered; retrying CSV load with latin1 encoding.")
            df_raw = pd.read_csv(csv_path, encoding="latin1")
        cols = df_raw.columns.tolist()
        date_col = st.selectbox("Date column", cols, index=cols.index("date") if "date" in cols else 0, key="inv_date_col")
        value_col = st.selectbox("Quantity column", cols, index=cols.index("quantity") if "quantity" in cols else min(1, len(cols)-1), key="inv_value_col")
        df = df_raw[[date_col, value_col]].rename(columns={date_col: "date", value_col: "quantity"})
    # Preprocess: daily demand and stats
    df["date"] = pd.to_datetime(df["date"])
    df["daily_demand"] = df["quantity"]
    avg_demand = df["daily_demand"].mean()
    std_demand = df["daily_demand"].std()
    lead_time = st.number_input("Lead time (days)", min_value=1, value=7)
    service_level = st.slider("Service level (%)", 50, 99, 95)
    z = norm.ppf(service_level / 100)
    reorder_point = avg_demand * lead_time + z * std_demand * (lead_time ** 0.5)
    d_annual = avg_demand * 365
    order_cost = st.number_input(
        "Order cost per order ($)", min_value=0.0, value=50.0
    )
    holding_cost = st.number_input(
        "Holding cost per unit per year ($)", min_value=0.0, value=2.0
    )
    EOQ = ((2 * d_annual * order_cost) / holding_cost) ** 0.5
    st.markdown(f"**Average daily demand:** {avg_demand:.2f}")
    st.markdown(f"**Std dev daily demand:** {std_demand:.2f}")
    st.markdown(f"**Reorder point:** {reorder_point:.2f}")
    st.markdown(f"**Economic Order Quantity (EOQ):** {EOQ:.2f}")

elif page == "Alerts":
    st.header("🚨 Alerts & Anomaly Detection")
    # Current stock level: manual entry or selecting from existing CSVs
    stock_source = st.radio(
        "Current stock source", ("Manual Entry", "Select from existing CSVs"),
        horizontal=True, key="alert_stock_source"
    )
    if stock_source == "Manual Entry":
        current_stock = st.number_input(
            "Current stock level", min_value=0.0, value=0.0, key="alert_current_stock"
        )
    else:
        # Choose from CSVs already uploaded under data/
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if not csv_files:
            st.warning(
                "No CSV files found in data directory. Please upload one on the Data Upload page."
            )
            st.stop()
        selected_csv = st.selectbox(
            "Select CSV for current stock", csv_files, key="alert_stock_select"
        )
        csv_path = os.path.join(DATA_DIR, selected_csv)
        # Load CSV with encoding fallback
        for enc in ("utf-8", "utf-8-sig", "latin1"):
            try:
                df_stock = pd.read_csv(csv_path, encoding=enc)
                if enc != "utf-8":
                    st.info(f"Loaded '{selected_csv}' with '{enc}' encoding due to decode issues.")
                break
            except Exception:
                continue
        else:
            st.error(
                "❌ Could not load stock CSV with encodings: utf-8, utf-8-sig, latin1"
            )
            st.stop()
        cols = df_stock.columns.tolist()
        stock_col = st.selectbox(
            "Select stock column", cols, key="alert_stock_col"
        )
        # Derive current stock from last numeric value
        stock_ser = pd.to_numeric(df_stock[stock_col], errors="coerce").dropna()
        if stock_ser.empty:
            st.error("Selected stock column has no numeric values.")
            st.stop()
        current_stock = stock_ser.iloc[-1]
        st.write(f"Using current stock level: {current_stock}")
    # Load demand data for alerts
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT date, quantity FROM demand ORDER BY date", conn
        )
        conn.close()
    except Exception:
        st.warning(
            "Could not load demand data.\n"
            "Please upload a CSV with 'date' and 'quantity' columns on the Data Upload page."
        )
        st.stop()
    avg_demand = df['quantity'].mean()
    std_demand = df['quantity'].std()
    lead_time = st.number_input(
        "Lead time for alerts (days)", min_value=1, value=7
    )
    service_level = st.slider(
        "Service level for alerts (%)", 50, 99, 95
    )
    z = norm.ppf(service_level / 100)
    rp = avg_demand * lead_time + z * std_demand * (lead_time ** 0.5)
    if current_stock < rp:
        st.warning(f"Stock below reorder point ({rp:.2f})! Consider replenishing.")
    else:
        st.success("Stock level is above reorder point.")
    st.markdown("---")
    st.subheader("Demand Anomalies (±3σ)")
    df['z_score'] = (df['quantity'] - avg_demand) / std_demand
    anomalies = df[abs(df['z_score']) > 3]
    if anomalies.empty:
        st.write("No anomalies detected.")
    else:
        st.write(anomalies[['date', 'quantity', 'z_score']])

elif page == "Dashboard":
    st.header("📊 KPI Dashboard")
    # Select dataset for KPI dashboard
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        st.warning("No CSV files found in data directory. Please upload one on the Data Upload page.")
        st.stop()
    selected_csv = st.selectbox("Select dataset for KPI Dashboard", csv_files)
    csv_path = os.path.join(DATA_DIR, selected_csv)
    # Load CSV with encoding fallback
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df_raw = pd.read_csv(csv_path, encoding=enc)
            if enc != "utf-8":
                st.info(f"Loaded '{selected_csv}' with '{enc}' encoding due to decode issues.")
            break
        except Exception:
            continue
    else:
        st.error("❌ Could not load CSV with encodings: utf-8, utf-8-sig, latin1")
        st.stop()
    st.subheader("Data Preview")
    st.dataframe(df_raw.head())
    st.subheader("Identifying KPIs")
    if st.button("Generate KPIs"):
        with st.spinner("Identifying KPIs..."):
            columns = df_raw.columns.tolist()
            prompt = (
                "You are a data analyst. "
                f"Given a dataset with the following columns: {columns}, "
                "list the top 5 key performance indicators (KPIs) relevant for this dataset. "
                "For each KPI, provide:\n"
                "1. KPI name\n"
                "2. Description\n"
                "3. How it could be calculated from the data (e.g., column names and calculation)\n\n"
                "Provide the response in JSON format with keys 'kpis' mapping to a list of objects "
                "with 'name', 'description', and 'calculation'."
            )
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that identifies KPIs from a dataset."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                content = response.choices[0].message.content or ""
                # Extract JSON from markdown code fences if present
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content, re.S)
                json_str = match.group(1) if match else content
                kpi_info = json.loads(json_str)
            except Exception as e:
                st.error(f"Error during KPI identification: {e}")
                st.write(content if 'content' in locals() else "")
                kpi_info = None
            if kpi_info and isinstance(kpi_info.get("kpis"), list):
                st.subheader("KPIs")
                for kpi in kpi_info["kpis"]:
                    name = kpi.get("name")
                    desc = kpi.get("description")
                    calc = kpi.get("calculation")
                    st.markdown(f"**{name}**: {desc}. Calculation: {calc}")
                st.markdown("---")
                st.subheader("KPI Dashboard Visualizations")
                if "date" in df_raw.columns:
                    df_raw["date"] = pd.to_datetime(df_raw["date"])
                for kpi in kpi_info["kpis"]:
                    name = kpi.get("name")
                    calc = kpi.get("calculation", "")
                    lower_calc = calc.lower()
                    if "sum" in lower_calc or "total" in lower_calc:
                        col = next((c for c in df_raw.columns if c.lower() in lower_calc), None)
                        if col and pd.api.types.is_numeric_dtype(df_raw[col]):
                            value = df_raw[col].sum()
                            st.metric(name, f"{value:.2f}")
                    elif "average" in lower_calc or "mean" in lower_calc:
                        col = next((c for c in df_raw.columns if c.lower() in lower_calc), None)
                        if col and pd.api.types.is_numeric_dtype(df_raw[col]):
                            value = df_raw[col].mean()
                            st.metric(name, f"{value:.2f}")
                    elif "trend" in lower_calc or "over time" in lower_calc:
                        col = next((c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c]) and c.lower() in lower_calc), None)
                        if "date" in df_raw.columns and col:
                            df_ts = df_raw.set_index("date")[col].resample("M").sum()
                            st.line_chart(df_ts, use_container_width=True)
    st.markdown("---")

elif page == "PO Management":
    st.header("🧾 Purchase Order Management")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS purchase_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sku TEXT,
            quantity REAL,
            order_date TEXT,
            expected_date TEXT,
            status TEXT
        )
        """
    )
    conn.commit()
    sku = st.text_input("SKU")
    qty = st.number_input("Quantity", min_value=0.0, value=0.0)
    order_date = st.date_input("Order date", datetime.today())
    expected_date = st.date_input("Expected arrival", datetime.today())
    status = st.selectbox("Status", ["Ordered", "Received", "Delayed"])
    if st.button("Create PO"):
        c.execute(
            "INSERT INTO purchase_orders (sku, quantity, order_date, expected_date, status) VALUES (?, ?, ?, ?, ?)",
            (
                sku,
                qty,
                order_date.isoformat(),
                expected_date.isoformat(),
                status,
            ),
        )
        conn.commit()
        st.success("Purchase order created.")
    st.subheader("Existing POs")
    pos = pd.read_sql_query(
        "SELECT * FROM purchase_orders ORDER BY order_date DESC", conn
    )
    st.dataframe(pos)
    conn.close()