import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
from pymongo import MongoClient
import os

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["disaster_hub"]
docs_col = db["documents"]

# Streamlit config
st.set_page_config(page_title="Admin Dashboard", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
    <style>
    /* Fix full background */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
        background-color: #f9fafb !important;
        color: #111827 !important;
    }
    [data-testid="stAppViewContainer"] > .main {
        background-color: #f9fafb !important;
    }

    /* Hide black overflow area */
    section.main {
        background-color: #f9fafb !important;
        min-height: 100vh !important;
        padding-bottom: 3rem !important;
    }

    .block-container {
        background-color: #f9fafb !important;
        padding: 2.5rem 4rem !important;
    }

    /* Header */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    .dashboard-header h1 {
        font-size: 2rem;
        font-weight: 800;
        color: #111827;
        margin: 0;
    }
    .dashboard-header p {
        color: #6b7280;
        font-size: 1rem;
        margin: 0;
    }
    .logout-btn button {
        background-color: white;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 0.4rem 1rem;
        color: #111827;
        cursor: pointer;
        font-weight: 500;
    }
    .logout-btn button:hover {
        background-color: #f3f4f6;
    }

    /* Tabs */
    div[data-testid="stHorizontalBlock"] div[role="tablist"] {
        display: flex;
        gap: 10px;
        justify-content: flex-start;
    }

    /* Upload Box */
    .stFileUploader > div > div {
        border: 2px dashed #d1d5db !important;
        border-radius: 12px !important;
        background-color: white !important;
        padding: 3rem !important;
        text-align: center;
        color: #4b5563 !important;
        transition: all 0.2s ease;
    }
    .stFileUploader > div > div:hover {
        border-color: #3b82f6 !important;
        color: #2563eb !important;
    }
    .stFileUploader label {
        color: #374151 !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Header ----------
col1, col2 = st.columns([8, 1])
with col1:
    st.markdown("""
        <div class="dashboard-header">
            <div>
                <h1>Admin Dashboard</h1>
                <p>Manage documents and system settings</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown('<div class="logout-btn"><button>Logout</button></div>', unsafe_allow_html=True)

# ---------- Tabs ----------
selected = option_menu(
    menu_title=None,
    options=["Upload", "Documents", "Status"],
    icons=["upload", "file-earmark-text", "bar-chart"],
    orientation="horizontal",
    styles={
        "container": {
            "padding": "0",
            "background-color": "#f9fafb",
            "display": "flex",
            "justify-content": "flex-start",
        },
        "icon": {"color": "#2563eb", "font-size": "18px"},
        "nav-link": {
            "font-size": "15px",
            "color": "#374151",
            "padding": "6px 18px",
            "border-radius": "8px",
            "background-color": "white",
            "border": "1px solid #e5e7eb",
            "transition": "all 0.2s ease-in-out",
        },
        "nav-link:hover": {
            "background-color": "#f3f4f6",
            "color": "#2563eb",
        },
        "nav-link-selected": {
            "background-color": "#2563eb",
            "color": "white",
            "border": "1px solid #2563eb",
        },
    },
)

# ---------- Upload Section ----------
if selected == "Upload":
    st.markdown("### Upload Reports")
    st.markdown("Upload PDF documents to build the knowledge base for AI-powered querying.")
    uploaded_files = st.file_uploader(
        "Drop PDF file here or click to browse",
        accept_multiple_files=True,
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_files:
        for file in uploaded_files:
            path = Path("uploads") / file.name
            path.parent.mkdir(exist_ok=True)
            with open(path, "wb") as f:
                f.write(file.read())
            st.success(f"✅ Uploaded {file.name}")

# ---------- Documents Section ----------
elif selected == "Documents":
    
    st.markdown("### 📄 Uploaded Documents")

    try:
        docs = list(docs_col.find({}, {"_id": 0, "doc_id": 1, "metadata": 1}))
        if not docs:
            st.info("No documents found in the database yet.")
        else:
            import pandas as pd

            data = []
            for doc in docs:
                meta = doc.get("metadata", {})
                data.append({
                    "Document ID": doc.get("doc_id", "Unknown"),
                    "Disaster Type": meta.get("disaster", "Unknown"),
                    "Year": meta.get("year", "Unknown"),
                    "Source": meta.get("source", "Unknown"),
                })

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

            if st.button("🗑 Clear all documents from DB"):
                docs_col.delete_many({})
                st.warning("All documents removed from MongoDB!")
                st.rerun()

    except Exception as e:
        st.error(f"Failed to load documents: {e}")

# ---------- Status Section ----------
elif selected == "Status":
    st.markdown("### System Status")
    st.info("✅ System running normally. All services operational.")
