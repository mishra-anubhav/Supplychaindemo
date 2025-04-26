# tools/sql_tool.py

import os
import sqlite3
import re
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

import openai

# 1️⃣ Load environment
load_dotenv()
import openai
openai.api_key = st.secrets["openai"]["api_key"]
openai.api_key = openaiApiKey
 

def _fallback_nl(raw_result: str, user_query: str = None) -> str:
    """
    Simple local fallback to convert raw SQL result into plain English.
    Tries to extract numeric or string values and produce a sentence.
    """
    # 1) Numeric answers
    nums = re.findall(r"-?\d+", raw_result)
    if nums:
        uniq_nums = list(dict.fromkeys(nums))
        if len(uniq_nums) == 1:
            return f"The answer is {uniq_nums[0]}."
        return f"The results are: {', '.join(uniq_nums)}."
    # 2) String answers (tuple format e.g. ('value',))
    vals = re.findall(r"\('([^']*)'", raw_result)
    if vals:
        uniq_vals = list(dict.fromkeys(vals))
        # Single value, attempt to mirror the question
        if len(uniq_vals) == 1 and user_query:
            # e.g. "What is the product type of SKU0?"
            uq = user_query.strip().rstrip('?')
            low = uq.lower()
            if " of " in low:
                before, after = uq.split(' of ', 1)
                attr = before.replace('What is the ', '').strip()
                key = after.strip()
                return f"The {attr} of {key} is {uniq_vals[0]}."
            return f"The result is {uniq_vals[0]}."
        # Multiple values
        return f"The results are: {', '.join(uniq_vals)}."
    # 3) Fallback: return raw
    return raw_result
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../database/supplychain.db"))
# 🔧 Ensure compatibility: create an 'inventory' view aliasing the 'demand' table
try:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP VIEW IF EXISTS inventory;")
    cursor.execute("CREATE VIEW inventory AS SELECT rowid AS id, * FROM demand;")
    conn.commit()
finally:
    conn.close()

# 2️⃣ Connect SQLite to LangChain
sql_db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# 3️⃣ Setup LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 4️⃣ Create SQL generator chain (LLM just generates SQL)
sql_generator_chain = create_sql_query_chain(llm, sql_db)

# 5️⃣ Function to run final SQL
def execute_sql(query: str) -> str:
    """
    Run a single SELECT statement directly against the SQLite DB.
    Sanitizes multiple statements and returns raw textual result.
    """
    # Sanitize: only use the first statement
    sql = query.strip().rstrip(';')
    if ';' in sql:
        sql = sql.split(';', 1)[0]
    # Execute via sqlite3 directly
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
    except Exception as e:
        return f"❌ SQL Execution Error: {e}"
    finally:
        try:
            conn.close()
        except:
            pass
    # Format result
    if not rows:
        return "No results."
    # Single scalar
    if len(rows) == 1 and len(rows[0]) == 1:
        return str(rows[0][0])
    # Multiple rows or columns
    return "\n".join([str(row) for row in rows])
    
def _natural_language_summary(user_query: str, sql_query: str, raw_result: str) -> str:
    """
    Use OpenAI LLM to turn the SQL result into a natural-language answer.
    """
    # If OpenAI API key is missing, use simple fallback
    if not openai.api_key:
        return _fallback_nl(raw_result, user_query)
    try:
        # Use new OpenAI Python v1.x API
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": (
                    "You are an assistant that converts SQL query results into a concise, natural language answer."
                )},
                {"role": "user", "content": (
                    f"User question: {user_query}\n"
                    f"SQL query: {sql_query}\n"
                    f"Raw result: {raw_result}\n"
                    "Provide a clear natural language response to the user's question."
                )},
            ],
            temperature=0,
        )
        ans = resp.choices[0].message.content
        # If model simply echoed tuples/lists, fall back to simpler summarizer
        if re.match(r"^[\[\(]", ans.strip()):
            return _fallback_nl(raw_result, user_query)
        return ans
    except Exception:
        return _fallback_nl(raw_result, user_query)

# 6️⃣ Full pipeline
def query_sql_database(user_query: str) -> str:
    try:
        # 🧠 Step 1: LLM generates SQL query
        sql_query = sql_generator_chain.invoke({"question": user_query})
        if isinstance(sql_query, dict):
            sql_query = sql_query.get("result", "")
        
        print("📝 SQL generated:", sql_query)

        # 🧠 Step 2: Execute SQL
        raw_result = execute_sql(sql_query)
        # 🧠 Step 3: Summarize result in natural language
        return _natural_language_summary(user_query, sql_query, raw_result)
    except Exception as e:
        return f"❌ Error: {e}"

# 7️⃣ Test
if __name__ == "__main__":
    question = "Which SKUs had demand above 500 in Q1?"
    print("🤖", query_sql_database(question))
