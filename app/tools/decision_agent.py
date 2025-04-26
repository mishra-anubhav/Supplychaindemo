# tools/decision_agent.py

import re

# Define keywords that suggest structured (SQL) queries
STRUCTURED_KEYWORDS = [
    r"\bforecast\b", r"\bquantity\b", r"\bq[1-4]\b",
    r"\baverage\b", r"\btotals?\b", r"\bSKU\b", r"\breorder\b",
    r"\binventory\b", r"\bstock\b", r"\bdemand\b"
]

def is_structured_question(question: str) -> bool:
    """
    Determines if a user question should be routed to SQL (structured) or Pinecone (unstructured)
    Returns True if structured (SQL), False otherwise.
    """
    question = question.lower()
    return any(re.search(pattern, question) for pattern in STRUCTURED_KEYWORDS)

# For quick testing
if __name__ == "__main__":
    test_questions = [
        "Forecast demand for SKU-123 in Q2",
        "What are the common supplier challenges?",
        "Show me average quantity sold",
        "Summarize the sustainability practices",
        "Which SKUs need reorder next month?"
    ]

    for q in test_questions:
        result = "SQL" if is_structured_question(q) else "Pinecone"
        print(f"🧠 '{q}' → {result}")

#Explanation
"""
What it does:
This utility file acts as a lightweight agent that analyzes user questions and decides:

Structured query (SQL) → if the question includes terms like "forecast", "Q1", "average", "SKU", etc.

Unstructured query (Pinecone) → everything else (e.g., policies, summaries, qualitative text)

⚙️ How it works:
Uses regex-based pattern matching for keywords (case-insensitive).

Easy to plug into your main app logic for routing.


"""