```python
# ============================================================
#                 RAG AI CHATBOT
#              STREAMLIT VERSION
#
#        GROQ + FAISS + TF-IDF + PDF
# ============================================================

import os
import re
import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f7f8;
    }

    .title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 17px;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. TITLE
# ============================================================

st.markdown(
    '<div class="title">🤖 RAG AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Chat with your PDF using Groq AI'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 4. HARDCODED GROQ API KEY
# ============================================================

GROQ_API_KEY = "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"


# ============================================================
# 5. GROQ CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# 6. MODEL
# ============================================================

MODEL_NAME = "llama-3.3-70b-versatile"


# ============================================================
# 7. SESSION STATE
# ============================================================

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []


if "vector_db" not in st.session_state:
    st.session_state.vector_db = None


if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None


if "document_name" not in st.session_state:
    st.session_state.document_name = ""


if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 8. CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.replace("\n", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 9. CREATE CHUNKS
# ============================================================

def create_chunks(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(
                chunk.strip()
            )

        start += chunk_size - overlap

    return chunks


# ============================================================
# 10. EXTRACT PDF TEXT
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(
        uploaded_file
    )

    complete_text = ""

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        page_text = page.extract_text()

        if page_text:

            complete_text += (
                f"\n[Page {page_number}]\n"
            )

            complete_text += page_text

    return clean_text(
        complete_text
    )


# ============================================================
# 11. CREATE FAISS VECTOR DATABASE
# ============================================================

def create_vector_database(chunks):

    vectorizer = TfidfVectorizer(

        stop_words="english",

        max_features=10000,

        ngram_range=(1, 2)

    )

    vectors = (
        vectorizer
        .fit_transform(chunks)
        .toarray()
    )

    vectors = vectors.astype(
        "float32"
    )

    faiss.normalize_L2(
        vectors
    )

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(vectors)

    return index, vectorizer


# ============================================================
# 12. PROCESS PDF
# ============================================================

def process_pdf(uploaded_file):

    try:

        text = extract_pdf_text(
            uploaded_file
        )

        if not text:

            return False, (
                "No readable text found "
                "in this PDF."
            )


        chunks = create_chunks(
            text
        )

        if not chunks:

            return False, (
                "Could not create document chunks."
            )


        index, vectorizer = (
            create_vector_database(
                chunks
            )
        )


        st.session_state.document_chunks = (
            chunks
        )

        st.session_state.vector_db = (
            index
        )

        st.session_state.tfidf_vectorizer = (
            vectorizer
        )

        st.session_state.document_name = (
            uploaded_file.name
        )


        # Reset chat for new document
        st.session_state.messages = []


        return True, (
            f"PDF processed successfully!\n\n"
            f"Document: {uploaded_file.name}\n\n"
            f"Chunks created: {len(chunks)}"
        )


    except Exception as e:

        return False, (
            f"PDF processing error:\n\n{str(e)}"
        )


# ============================================================
# 13. RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(
    question,
    top_k=5
):

    vector_db = (
        st.session_state.vector_db
    )

    vectorizer = (
        st.session_state.tfidf_vectorizer
    )

    chunks = (
        st.session_state.document_chunks
    )


    if vector_db is None:

        return []


    question_vector = (
        vectorizer
        .transform([question])
        .toarray()
    )


    question_vector = (
        question_vector
        .astype("float32")
    )


    faiss.normalize_L2(
        question_vector
    )


    scores, indexes = (
        vector_db.search(
            question_vector,
            min(
                top_k,
                len(chunks)
            )
        )
    )


    results = []


    for score, index in zip(
        scores[0],
        indexes[0]
    ):

        if index >= 0:

            results.append({

                "text":
                    chunks[index],

                "score":
                    float(score)

            })


    return results


# ============================================================
# 14. GENERATE ANSWER
# ============================================================

def generate_answer(question):

    results = retrieve_chunks(
        question,
        top_k=5
    )


    if not results:

        return (
            "The answer is not available "
            "in the uploaded document."
        )


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    context = ""

    for i, result in enumerate(
        results,
        start=1
    ):

        context += (

            f"\n\n"
            f"========== CONTEXT {i} ==========\n"
            f"{result['text']}"

        )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    system_prompt = """

You are a professional RAG AI assistant.

Answer questions using the uploaded
document context.

Rules:

1. Use the document context as your
   primary source.

2. Do not invent information.

3. If the answer cannot be found
   in the document, say:

   "The answer is not available
   in the uploaded document."

4. Give clear and concise answers.

5. Use bullet points when useful.

6. Mention page numbers when
   available.

"""


    # ========================================================
    # USER PROMPT
    # ========================================================

    user_prompt = f"""

DOCUMENT CONTEXT:

{context}


USER QUESTION:

{question}


Answer the question using the
document context.
"""


    # ========================================================
    # GROQ MESSAGES
    # ========================================================

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]


    # Add previous conversation
    for message in st.session_state.messages:

        messages.append({

            "role":
                message["role"],

            "content":
                message["content"]

        })


    # Current question
    messages.append({

        "role":
            "user",

        "content":
            user_prompt

    })


    # ========================================================
    # GROQ API
    # ========================================================

    try:

        response = (
            client
            .chat
            .completions
            .create(

                model=MODEL_NAME,

                messages=messages,

                temperature=0.3,

                max_completion_tokens=1024

            )
        )


        answer = (
            response
            .choices[0]
            .message
            .content
        )


    except Exception as e:

        answer = (
            f"Groq API Error:\n\n{str(e)}"
        )


    # ========================================================
    # SOURCES
    # ========================================================

    answer += "\n\n---\n**Sources used:**\n"

    for i, result in enumerate(
        results[:3],
        start=1
    ):

        answer += (

            f"- Context {i} "
            f"(similarity: "
            f"{result['score']:.2f})\n"

        )


    return answer


# ============================================================
# 15. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Document")


    uploaded_file = st.file_uploader(

        "Upload your PDF",

        type=["pdf"]

    )


    if uploaded_file:

        if st.button(
            "Process PDF",
            use_container_width=True
        ):

            with st.spinner(
                "Processing PDF..."
            ):

                success, message = (
                    process_pdf(
                        uploaded_file
                    )
                )


            if success:

                st.success(message)

            else:

                st.error(message)


    # ========================================================
    # DOCUMENT STATUS
    # ========================================================

    if st.session_state.document_name:

        st.divider()

        st.success(
            "📄 "
            + st.session_state.document_name
        )

        st.write(
            "🧩 Chunks: "
            + str(
                len(
                    st.session_state.document_chunks
                )
            )
        )


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    st.divider()


    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# 16. MAIN CHAT
# ============================================================

if not st.session_state.document_chunks:

    st.info(
        "👈 Upload a PDF from the sidebar "
        "and click **Process PDF** to start."
    )


# ============================================================
# 17. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 18. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your document..."
)


if question:

    # --------------------------------------------------------
    # Check document
    # --------------------------------------------------------

    if not st.session_state.document_chunks:

        st.warning(
            "Please upload and process "
            "a PDF first."
        )

        st.stop()


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # Save user message
    st.session_state.messages.append({

        "role":
            "user",

        "content":
            question

    })


    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            answer = generate_answer(
                question
            )

        st.markdown(
            answer
        )


    # Save assistant message
    st.session_state.messages.append({

        "role":
            "assistant",

        "content":
            answer

    })
```
