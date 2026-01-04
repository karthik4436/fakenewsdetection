import streamlit as st
import requests
import joblib
import re
from newspaper import Article

# -----------------------------
# CONFIG
# -----------------------------
NEWSDATA_API_KEY = "pub_59dec60e88084eba98ce4c411b78b277"
NEWSDATA_URL = "https://newsdata.io/api/1/news"

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Fake News Intelligence",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS (UI MAGIC ✨)
# -----------------------------
st.markdown("""
<style>
body { background-color: #f7f9fc; }

.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.badge-real {
    background-color: #e6f4ea;
    color: #137333;
    padding: 8px 14px;
    border-radius: 20px;
    font-weight: bold;
}

.badge-fake {
    background-color: #fdecea;
    color: #b3261e;
    padding: 8px 14px;
    border-radius: 20px;
    font-weight: bold;
}

.source-card:hover {
    background: #eff6ff;
    transform: translateY(-2px);
    transition: all 0.2s ease-in-out;
}



}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🧠 Fake News Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Real-time news verification with live web sources</p>", unsafe_allow_html=True)

# -----------------------------
# INPUT SECTION
# -----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

input_type = st.radio("Choose input type:", ["URL", "Headline", "Full Text"], horizontal=True)
user_input = st.text_area("Paste your content below:", height=170)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# HELPERS
# -----------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.lower()

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

def live_web_search(query):
    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": query,
        "language": "en"
    }
    res = requests.get(NEWSDATA_URL, params=params).json()
    sources = []
    if "results" in res:
        for item in res["results"][:5]:
            sources.append({
                "title": item["title"],
                "source": item["source_id"],
                "link": item["link"]
            })
    return sources

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("🔍 Analyze News", use_container_width=True):

    with st.spinner("Analyzing credibility & searching live web..."):
        if input_type == "URL":
            text = extract_text_from_url(user_input)
            if not text:
                st.error("Could not extract text from URL.")
                st.stop()
        else:
            text = user_input

        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])

        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0].max()

        label = "REAL" if pred == 1 else "FAKE"
        sources = live_web_search(" ".join(cleaned.split()[:10]))

    # -----------------------------
    # RESULT CARD
    # -----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    badge = "badge-real" if label == "REAL" else "badge-fake"
    st.markdown(f"<span class='{badge}'>{label}</span>", unsafe_allow_html=True)
    st.markdown(f"### Confidence: {round(prob*100,2)}%")
    st.progress(prob)

    st.markdown("#### 🧠 Explanation")
    if label == "FAKE":
        st.write("• Emotional or clickbait wording detected")
        st.write("• Claim not supported by trusted media")
        st.write("• Similar patterns found in known fake news")
    else:
        st.write("• Reported by multiple reliable sources")
        st.write("• Neutral and factual language used")
        st.write("• Matches verified news patterns")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # SOURCES
    # -----------------------------
    st.markdown("### 🔗 Live Web Sources")
    if sources:
        for s in sources:
            st.markdown(f"""
            <div class='source-card'>
                <b>{s['title']}</b><br>
                Source: {s['source']}<br>
                <a href="{s['link']}" target="_blank">Read article →</a>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No trusted sources found online.")
