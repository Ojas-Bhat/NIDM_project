import streamlit as st
import os
import time
from pymongo import MongoClient, errors
from cb import retrieve_relevant_chunks, generate_answer
from datapop3 import extract_text_from_pdf, clean_and_chunk_text, extract_metadata, store_document

# ------------------------------
# Password hashing helpers
# ------------------------------
import hashlib
try:
    import bcrypt
    HAS_BCRYPT = True
except Exception:
    HAS_BCRYPT = False

def hash_password(password: str) -> dict:
    """Hash password securely."""
    if HAS_BCRYPT:
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return {"algo": "bcrypt", "hash": hashed.decode("utf-8")}
    else:
        salt = os.urandom(16).hex()
        hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
        return {"algo": "sha256", "salt": salt, "hash": hashed}

def verify_password(password: str, stored: dict) -> bool:
    algo = stored.get("algo", "sha256")
    if algo == "bcrypt" and HAS_BCRYPT:
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored["hash"].encode("utf-8"))
        except Exception:
            return False
    elif algo == "sha256":
        salt = stored.get("salt", "")
        expected = stored.get("hash", "")
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest() == expected
    return False

# ------------------------------
# MongoDB setup
# ------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["disaster_hub"]
docs_col = db["documents"]
users_col = db["users"]

try:
    users_col.create_index("email", unique=True)
except errors.PyMongoError:
    pass

# ------------------------------
# Streamlit config
# ------------------------------
st.set_page_config(page_title="NIDM Disaster Knowledge Chatbot", layout="wide", initial_sidebar_state="expanded")

# ------------------------------
# Session management
# ------------------------------
if "auth" not in st.session_state:
    st.session_state.auth = False
if "user" not in st.session_state:
    st.session_state.user = None
if "role" not in st.session_state:
    st.session_state.role = None

# ------------------------------
# Auth UI
# ------------------------------
def login_signup_screen():
    st.title("🔐 Welcome to NIDM Disaster Knowledge Chatbot")

    tabs = st.tabs(["Login", "Sign Up"])

    # ---- Login Tab ----
    with tabs[0]:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Sign In"):
            if not email or not password:
                st.warning("Please enter both email and password.")
            else:
                user = users_col.find_one({"email": email.strip().lower()})
                if user and verify_password(password, user.get("password", {})):
                    st.session_state.auth = True
                    st.session_state.user = {
                        "name": user.get("name"),
                        "email": user.get("email")
                    }
                    st.session_state.role = user.get("role", "User")
                    st.success(f"Welcome back, {user.get('name','User')}! Redirecting…")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

    # ---- Sign Up Tab ----
    with tabs[1]:
        st.subheader("Create an account")
        name = st.text_input("Full Name", key="signup_name")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
        role = st.selectbox("Choose role", ["User", "Admin"], help="Admins can upload/manage PDFs.")
        admin_invite = st.text_input("Admin Invite Code (required for Admin role)", type="password") if role == "Admin" else ""

        if st.button("Create Account"):
            if not name or not email or not password or not confirm:
                st.warning("Please fill all fields.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif role == "Admin":
                required_code = os.environ.get("ADMIN_INVITE_CODE", "1234")
                if required_code and admin_invite != required_code:
                    st.error("Invalid Admin Invite Code.")
                    st.stop()
            try:
                pwd = hash_password(password)
                doc = {
                    "name": name.strip(),
                    "email": email.strip().lower(),
                    "role": role,
                    "password": pwd,
                }
                users_col.insert_one(doc)
                st.success("Account created! You can sign in now.")
                st.balloons()
            except errors.DuplicateKeyError:
                st.error("An account with this email already exists.")
            except Exception as e:
                st.error(f"Could not create account: {e}")

# ------------------------------
# Main App
# ------------------------------
def main_app():
    st.title("NIDM Disaster Knowledge Chatbot")

    with st.sidebar:
        st.write(f"*Signed in as:* {st.session_state.user.get('name','')} ({st.session_state.user.get('email','')})")
        st.write(f"*Role:* {st.session_state.role}")
        if st.button("🚪 Log out"):
            st.session_state.auth = False
            st.session_state.user = None
            st.session_state.role = None
            st.success("Logged out.")
            time.sleep(0.5)
            st.rerun()

    # ---- Admin Panel ----
    if st.session_state.role == "Admin":
        st.markdown("<h1 style='font-size:40px;'>Admin Dashboard</h1>", unsafe_allow_html=True)
        st.caption("Manage documents and system settings")

        tabs = st.tabs(["📤 Upload", "📑 Documents", "📊 Status"])

        # ---- Upload Tab ----
        with tabs[0]:
            st.subheader("Upload Reports")
            st.markdown("Upload PDF documents to build the knowledge base for AI-powered querying")
            uploaded_file = st.file_uploader(" ", type=["pdf"], label_visibility="collapsed")

            if uploaded_file:
                save_path = os.path.join("data", uploaded_file.name)
                os.makedirs("data", exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ {uploaded_file.name} uploaded successfully!")

                if st.button("Process & Add to Database"):
                    with st.spinner("Extracting and embedding..."):
                        try:
                            text = extract_text_from_pdf(save_path)
                            chunks = clean_and_chunk_text(text)
                            metadata = extract_metadata(text, uploaded_file.name)
                            store_document(uploaded_file.name, chunks, metadata)
                            st.success("Document processed and added to MongoDB.")
                        except Exception as e:
                            st.error(f"Processing failed: {e}")

            if st.button("🗑 Clear all documents"):
                try:
                    docs_col.delete_many({})
                    st.warning("All documents removed from MongoDB!")
                except Exception as e:
                    st.error(f"Failed to clear documents: {e}")

        # ---- Documents Tab ----
        with tabs[1]:
            st.subheader("📄 Uploaded Documents")

            uploaded_docs = list(docs_col.find({}, {"_id": 0, "filename": 1, "metadata": 1}))

            if not uploaded_docs:
                st.info("No uploaded reports found.")
            else:
                for doc in uploaded_docs:
                    filename = doc.get("filename", "Unnamed File")
                    metadata = doc.get("metadata", {})
                    st.markdown(f"""
                        <div style="background-color:#f9fafb; border-radius:8px; padding:12px; margin-bottom:8px; border:1px solid #e5e7eb;">
                            <strong>📘 {filename}</strong><br>
                            <small style="color:#6b7280;">
                                {metadata.get('source','Unknown source')}<br>
                                Uploaded: {metadata.get('uploaded_at','N/A')}
                            </small>
                        </div>
                    """, unsafe_allow_html=True)

        # ---- Status Tab ----
        with tabs[2]:
            st.subheader("📈 System Status")
            total_docs = docs_col.count_documents({})
            total_users = users_col.count_documents({})
            st.metric("Total Documents", total_docs)
            st.metric("Total Users", total_users)
            st.success("System operational ✅")

    # ---- User Panel ----
    else:
        st.markdown("<h1 style='font-size:32px; margin-bottom:0;'>Intelligence Dashboard</h1>", unsafe_allow_html=True)
        st.caption("Query reports and visualize insights")

        tabs = st.tabs(["💬 Chat", "📊 Visualizations"])

        with tabs[0]:
            st.markdown("""
                <div style="text-align:center; margin-top:60px;">
                    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712100.png" width="75">
                    <h3 style="margin-top:15px; margin-bottom:8px;">Start a conversation</h3>
                    <p style="color:#6b7280; font-size:15px;">
                        Ask questions about the uploaded disaster reports. I'll use AI and RAG to provide accurate answers based on the documents.
                    </p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            user_query = st.text_input("Ask about disaster reports...", key="user_query", label_visibility="collapsed")
            col1, col2 = st.columns([8, 1])
            with col2:
                send = st.button("📤 Send", key="send_btn")

            if send and user_query.strip():
                with st.spinner("Retrieving relevant information..."):
                    try:
                        retrieved = retrieve_relevant_chunks(user_query)
                        if not retrieved:
                            st.warning("No relevant documents found.")
                        else:
                            answer = generate_answer(user_query, retrieved)
                            st.markdown(f"""
                                <div style='margin-top:20px; padding:20px; border-radius:10px; background-color:#f3f4f6;'>
                                    <strong>🤖 Response:</strong><br>{answer}
                                
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Something went wrong while generating the answer: {e}")

        with tabs[1]:
            st.markdown("<h3 style='margin-top:30px;'>📊 Visualizations</h3>", unsafe_allow_html=True)
            st.info("Visual insights based on uploaded disaster data will appear here soon.")

# ------------------------------
# App Router
# ------------------------------
if not st.session_state.auth:
    login_signup_screen()
else:
    main_app()
