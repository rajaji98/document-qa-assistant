import os
import re
import math
import json
import hashlib
from collections import Counter

import streamlit as st
import pymupdf
import numpy as np
import faiss

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

load_dotenv()

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1200px;}
    .hero {padding: 1.25rem 1.4rem; border-radius: 18px; border: 1px solid rgba(128,128,128,.25); margin-bottom: 1rem; background: linear-gradient(120deg, rgba(80,120,200,.12), rgba(70,180,160,.08));}
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
    .hero p {margin: 0; opacity: .82;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: .8rem; border-radius: 12px;}
    div[data-testid="stChatMessage"] {border-radius: 14px; padding: .6rem .8rem;}
</style>
<div class="hero">
  <h1>📄 Document Q&amp;A Assistant</h1>
  <p>Ask questions across your PDFs and text files using hybrid retrieval, grounded Gemini answers, and source citations.</p>
</div>
""", unsafe_allow_html=True)

# Retrieval settings. Tune these after evaluating on your own documents.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
MIN_SEMANTIC_SCORE = 0.28
SEMANTIC_WEIGHT = 0.65
BM25_WEIGHT = 0.35


# --------------------------------------------------
# 2. CACHED MODELS / CLIENT
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_gemini_client(api_key):
    return genai.Client(api_key=api_key)


# --------------------------------------------------
# 3. SESSION STATE
# --------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_signature" not in st.session_state:
    st.session_state.document_signature = None


def clear_chat():
    st.session_state.chat_history = []


# --------------------------------------------------
# 4. EXTRACT TEXT FROM PDF / TXT
# --------------------------------------------------
def extract_documents(files):
    pages = []

    for uploaded_file in files:
        filename = uploaded_file.name
        content = uploaded_file.getvalue()

        if filename.lower().endswith(".pdf"):
            with pymupdf.open(stream=content, filetype="pdf") as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        pages.append({
                            "source": filename,
                            "page": page_number,
                            "text": text,
                        })

        elif filename.lower().endswith(".txt"):
            text = content.decode("utf-8", errors="replace").strip()
            if text:
                pages.append({
                    "source": filename,
                    "page": None,
                    "text": text,
                })

    return pages


# --------------------------------------------------
# 5. SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------
def chunk_pages(pages, chunk_size=120, overlap=20):
    """
    Create section-aware chunks while preserving source/page metadata.

    Recognized resume-style headings are separated even when PDF extraction
    places them on the same line. Other documents fall back to page-level
    chunking. Exact and highly similar duplicate chunks are filtered out.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    heading_names = [
        "PROFESSIONAL SUMMARY",
        "TECHNICAL SKILLS",
        "PROJECTS",
        "RK Krishna Classes",
        "SMS Spam Classifier",
        "Titanic Survival Prediction",
        "EDUCATION",
        "ACHIEVEMENTS",
        "CERTIFICATIONS",
        "EXPERIENCE",
        "INTERNSHIPS",
    ]
    heading_pattern = re.compile(
        r"(?i)(?<![A-Za-z])(" +
        "|".join(re.escape(name) for name in sorted(heading_names, key=len, reverse=True)) +
        r")(?![A-Za-z])"
    )

    chunks = []
    for page in pages:
        raw_text = re.sub(r"\s+", " ", page.get("text", "")).strip()
        if not raw_text:
            continue

        # Add split points before known headings that may be fused to preceding text.
        normalized = heading_pattern.sub(r"\n\1\n", raw_text)
        blocks = [part.strip() for part in re.split(r"\n+", normalized) if part.strip()]

        # Merge short non-heading fragments into the previous section to avoid
        # creating low-information chunks from PDF bullet/layout artifacts.
        sections = []
        current_heading = "Document"
        for block in blocks:
            matched = heading_pattern.fullmatch(block.strip())
            if matched:
                current_heading = matched.group(1).strip()
                sections.append({"heading": current_heading, "text": ""})
            else:
                if not sections:
                    sections.append({"heading": current_heading, "text": block})
                else:
                    if sections[-1]["text"]:
                        sections[-1]["text"] += " "
                    sections[-1]["text"] += block

        for section in sections:
            words = section["text"].split()
            if not words:
                continue

            # Keep section title with its content to improve retrieval context.
            heading = section["heading"]
            content_words = words
            if heading.lower() != "document":
                prefix = heading + ":"
                content_words = (prefix + " " + " ".join(words)).split()

            start_idx = 0
            while start_idx < len(content_words):
                end_idx = min(start_idx + chunk_size, len(content_words))
                chunk_text = " ".join(content_words[start_idx:end_idx]).strip()

                if chunk_text:
                    chunks.append({
                        "source": page["source"],
                        "page": page["page"],
                        "section": heading,
                        "text": chunk_text,
                    })

                if end_idx == len(content_words):
                    break
                start_idx = end_idx - overlap

    # Filter exact duplicates and very near-duplicates while retaining the
    # first/most complete occurrence. A conservative threshold avoids removing
    # chunks that share a heading but contain different project details.
    unique_chunks = []
    exact_seen = set()
    token_sets = []
    for chunk in chunks:
        normalized_text = re.sub(r"\W+", " ", chunk["text"].lower()).strip()
        if not normalized_text or normalized_text in exact_seen:
            continue

        current_tokens = set(normalized_text.split())
        is_near_duplicate = False
        for previous_tokens in token_sets:
            if not current_tokens or not previous_tokens:
                continue
            union = current_tokens | previous_tokens
            jaccard = len(current_tokens & previous_tokens) / len(union)
            if jaccard >= 0.90:
                is_near_duplicate = True
                break

        if not is_near_duplicate:
            exact_seen.add(normalized_text)
            token_sets.append(current_tokens)
            unique_chunks.append(chunk)

    return unique_chunks


# --------------------------------------------------
# 6. CREATE AND CACHE FAISS INDEX
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def create_search_index_cached(texts_tuple, embedding_model_name=EMBEDDING_MODEL_NAME):
    model = load_embedding_model()
    texts = list(texts_tuple)

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


# --------------------------------------------------
# 7. HYBRID RETRIEVAL: SEMANTIC + BM25
# --------------------------------------------------
def tokenize(text):
    """Simple lowercase word tokenizer for BM25 keyword search."""
    return re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", text.lower())


def bm25_scores(query, chunks, k1=1.5, b=0.75):
    """
    Lightweight BM25 implementation to avoid an extra dependency.
    Returns one BM25 relevance score per chunk.
    """
    tokenized_docs = [tokenize(chunk["text"]) for chunk in chunks]
    query_terms = tokenize(query)

    if not tokenized_docs or not query_terms:
        return [0.0] * len(chunks)

    doc_lengths = [len(doc) for doc in tokenized_docs]
    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)
    document_frequency = Counter()

    for doc in tokenized_docs:
        for term in set(doc):
            document_frequency[term] += 1

    n_docs = len(tokenized_docs)
    scores = []

    for doc, doc_length in zip(tokenized_docs, doc_lengths):
        frequencies = Counter(doc)
        score = 0.0

        for term in query_terms:
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue

            df = document_frequency.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (
                1 - b + b * doc_length / max(avgdl, 1)
            )
            score += idf * (tf * (k1 + 1)) / denominator

        scores.append(float(score))

    return scores


def retrieve_chunks(
    question,
    chunks,
    index,
    embedding_model,
    top_k=TOP_K,
    min_semantic_score=MIN_SEMANTIC_SCORE,
    semantic_weight=SEMANTIC_WEIGHT,
    bm25_weight=BM25_WEIGHT,
):
    """
    Combine normalized semantic similarity and normalized BM25 scores.
    A candidate is retained if it passes the semantic threshold OR
    has a positive keyword match. The final ranking uses hybrid score.
    """
    if not chunks:
        return []

    query_vector = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_vector = np.asarray(query_vector, dtype="float32")

    # Search all chunks so keyword matching and semantic ranking can be combined.
    semantic_scores, semantic_indices = index.search(
        query_vector, len(chunks)
    )

    semantic_by_index = {}
    for score, idx in zip(semantic_scores[0], semantic_indices[0]):
        if idx >= 0:
            semantic_by_index[int(idx)] = float(score)

    raw_bm25 = bm25_scores(question, chunks)
    max_bm25 = max(raw_bm25) if raw_bm25 else 0.0
    normalized_bm25 = [
        (score / max_bm25 if max_bm25 > 0 else 0.0)
        for score in raw_bm25
    ]

    candidates = []
    for idx, chunk in enumerate(chunks):
        semantic_score = semantic_by_index.get(idx, -1.0)
        bm25_score = normalized_bm25[idx]

        # Threshold guards against purely irrelevant semantic matches.
        # Exact keyword matches can still be retained even if semantic
        # similarity is lower than the threshold.
        if semantic_score < min_semantic_score and bm25_score <= 0:
            continue

        hybrid_score = (
            semantic_weight * max(semantic_score, 0.0)
            + bm25_weight * bm25_score
        )

        item = chunk.copy()
        item["similarity"] = semantic_score
        item["bm25_score"] = raw_bm25[idx]
        item["hybrid_score"] = hybrid_score
        candidates.append(item)

    candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)

    # Retrieval-time duplicate filtering: overlapping chunks can share much of
    # the same text even when they are not identical. Prefer the higher-ranked
    # passage and retain chunks with materially different content.
    deduplicated = []
    selected_token_sets = []
    for item in candidates:
        normalized_text = re.sub(r"\W+", " ", item["text"].lower()).strip()
        token_set = set(normalized_text.split())
        duplicate = False

        for previous_tokens in selected_token_sets:
            if not token_set or not previous_tokens:
                continue
            union = token_set | previous_tokens
            jaccard = len(token_set & previous_tokens) / len(union)
            if jaccard >= 0.82:
                duplicate = True
                break

        if not duplicate:
            deduplicated.append(item)
            selected_token_sets.append(token_set)

        if len(deduplicated) >= top_k:
            break

    return deduplicated


# --------------------------------------------------
# 8. REWRITE FOLLOW-UP QUESTIONS
# --------------------------------------------------
def rewrite_query(client, question, chat_history):
    """Convert a follow-up into a standalone query for document retrieval."""
    if not chat_history:
        return question

    model_name = os.getenv("GEMINI_MODEL")
    if not model_name:
        return question

    recent_history = chat_history[-6:]
    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in recent_history
    )

    prompt = f"""
You rewrite questions for a document search system.

Turn the latest question into a standalone search query using
conversation history.

Rules:
1. Preserve the user's intent and important technical terms.
2. Resolve references such as "it", "that", "the second one",
   and "explain more" when the history makes them clear.
3. Do not answer the question.
4. Do not invent names or facts.
5. If uncertain, keep the query close to the original.
6. Return only the rewritten query.

Conversation history:
{history_text}

Latest question:
{question}

Standalone search query:
"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        rewritten = (response.text or "").strip()
        if not rewritten or len(rewritten) > 1000:
            return question
        return rewritten
    except Exception:
        # Retrieval can continue with the original query if rewriting fails.
        return question


# --------------------------------------------------
# 9. GENERATE GROUNDED ANSWER WITH SOURCE LABELS
# --------------------------------------------------
def generate_answer(client, question, retrieved_chunks, chat_history):
    model_name = os.getenv("GEMINI_MODEL")
    if not model_name:
        raise ValueError("GEMINI_MODEL is missing from your .env file.")

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        location = chunk["source"]
        if chunk["page"] is not None:
            location += f", page {chunk['page']}"

        context_parts.append(
            f"[S{i}] Source: {location}\n"
            f"Content:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)
    recent_messages = chat_history[-6:]
    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in recent_messages
    )

    prompt = f"""
You are a precise document question-answering assistant.

Answer the current question using the retrieved document context.
Conversation history can help resolve follow-up references, but it is
not evidence for facts about the documents.

Rules:
1. Answer the exact question directly.
2. Use the retrieved document context as the factual source.
3. Include relevant details that are actually present in the context.
4. Do not invent facts, technologies, dates, URLs, or citations.
5. Cite factual claims using source labels such as [S1] and [S2].
6. If the context does not contain the answer, say that you could not
   find the information in the uploaded documents.
7. Do not claim that a source supports a statement unless it does.
8. Use clear headings and bullet points where useful.

Recent conversation:
{history_text if history_text else "No previous conversation."}

Retrieved document context:
{context}

Current question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    answer = (response.text or "").strip()
    if not answer:
        raise ValueError("Gemini returned an empty response.")
    return answer



# --------------------------------------------------
# EVALUATION HELPERS
# --------------------------------------------------
def evaluate_retrieval(test_cases, chunks, index, embedding_model, top_k=TOP_K):
    """Evaluate hybrid retrieval against user-labelled source/keyword expectations."""
    rows = []
    for case in test_cases:
        question = str(case.get("question", "")).strip()
        if not question:
            continue
        expected_source = str(case.get("expected_source", "")).strip().lower()
        expected_keywords = case.get("expected_keywords", [])
        if isinstance(expected_keywords, str):
            expected_keywords = [expected_keywords]
        expected_keywords = [
            str(k).strip().lower() for k in expected_keywords if str(k).strip()
        ]

        found = retrieve_chunks(
            question, chunks, index, embedding_model, top_k=top_k
        )

        def matches(chunk):
            source_match = bool(
                expected_source and expected_source in chunk["source"].lower()
            )
            text = chunk["text"].lower()
            keyword_match = any(keyword in text for keyword in expected_keywords)
            if expected_source and expected_keywords:
                return source_match and keyword_match
            return source_match or keyword_match

        ranks = [
            rank for rank, chunk in enumerate(found, start=1) if matches(chunk)
        ]
        first_rank = ranks[0] if ranks else None
        rows.append({
            "question": question,
            "expected_source": case.get("expected_source", ""),
            "expected_keywords": ", ".join(expected_keywords),
            "retrieved_count": len(found),
            "hit": bool(ranks),
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
            "precision_at_k": len(ranks) / max(len(found), 1),
            "top_sources": "; ".join(
                c["source"] + (f' p.{c["page"]}' if c["page"] is not None else "")
                for c in found[:3]
            ),
            "top_passage_preview": found[0]["text"][:260] if found else
                "No passages retrieved",
        })
    return rows


DEFAULT_EVAL_JSON = """[
  {
    "question": "Replace with a question answered by your document",
    "expected_source": "your-file.pdf",
    "expected_keywords": ["a distinctive phrase"]
  }
]"""


# --------------------------------------------------
# 10. UPLOAD DOCUMENTS
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "📁 Add your documents",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="You can upload multiple PDF and TXT files. Text-based PDFs work best; scanned PDFs may require OCR.",
)

if not uploaded_files:
    st.info("Upload at least one document to begin.")
    st.stop()

document_signature = tuple(
    sorted(
        (
            file.name,
            len(file.getvalue()),
            hashlib.sha256(file.getvalue()).hexdigest(),
        )
        for file in uploaded_files
    )
)

if (
    st.session_state.document_signature is not None
    and st.session_state.document_signature != document_signature
):
    st.session_state.chat_history = []

st.session_state.document_signature = document_signature


# --------------------------------------------------
# 11. PROCESS DOCUMENTS AND BUILD INDEX
# --------------------------------------------------
try:
    with st.spinner("Extracting document text..."):
        pages = extract_documents(uploaded_files)

    chunks = chunk_pages(pages, chunk_size=150, overlap=30)

    if not chunks:
        st.warning(
            "No extractable text was found. If your PDF contains scanned "
            "pages, OCR may be required."
        )
        st.stop()

    with st.spinner("Loading embedding model and indexing documents..."):
        embedding_model = load_embedding_model()
        texts_tuple = tuple(chunk["text"] for chunk in chunks)
        index = create_search_index_cached(
            texts_tuple,
            EMBEDDING_MODEL_NAME,
        )

except Exception as error:
    st.error("Could not process the uploaded documents.")
    st.caption(f"Technical details: {error}")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Files uploaded", len(uploaded_files))
with col2:
    st.metric("Pages / text sections", len(pages))
with col3:
    st.metric("Search chunks", len(chunks))

st.success("Document search index is ready! ✅")

# Main-panel navigation keeps evaluation easy to find, instead of hiding it
# behind a sidebar-only toggle.
workspace_view = st.radio(
    "Choose workspace",
    options=["💬 Chat with documents", "🧪 Evaluate retrieval & inspect chunks"],
    horizontal=True,
    key="workspace_view",
)
show_evaluation = workspace_view == "🧪 Evaluate retrieval & inspect chunks"

with st.sidebar:
    st.header("⚙️ Workspace")
    st.caption("Upload documents in the main panel. Evaluation uses the same active document set.")
    st.divider()
    st.markdown("**Current retrieval settings**")
    st.write(f"Top K: `{TOP_K}`")
    st.write(f"Semantic threshold: `{MIN_SEMANTIC_SCORE:.2f}`")
    st.write(
        f"Hybrid weights: `{SEMANTIC_WEIGHT:.0%}` semantic / "
        f"`{BM25_WEIGHT:.0%}` BM25"
    )
    st.caption("Use evaluation results to guide retrieval tuning.")

if show_evaluation:
    # Diagnostic view is intentionally available before the evaluation form.
    st.subheader("🔍 Inspect extracted text and indexed chunks")
    total_chars = sum(len(page["text"]) for page in pages)
    total_words = sum(len(page["text"].split()) for page in pages)
    diag_cols = st.columns(3)
    diag_cols[0].metric("Extracted text characters", f"{total_chars:,}")
    diag_cols[1].metric("Extracted words", f"{total_words:,}")
    diag_cols[2].metric("Indexed chunks", len(chunks))

    if total_words < 80:
        st.warning(
            "Very little text was extracted. If the PDF visibly contains more text, "
            "it may be scanned or have a complex layout. Check the previews below; "
            "OCR may be needed if the text is missing."
        )

    with st.expander("📄 View extracted text by page/section", expanded=True):
        if not pages:
            st.error("No text was extracted from the uploaded files.")
        for page_i, page in enumerate(pages, start=1):
            page_label = page["source"]
            if page["page"] is not None:
                page_label += f" — Page {page['page']}"
            st.markdown(f"**Extracted section {page_i}: {page_label}**")
            st.caption(
                f'{len(page["text"].split())} words · {len(page["text"])} characters'
            )
            st.text_area(
                f"Extracted text {page_i}",
                value=page["text"],
                height=180,
                key=f"diagnostic_extracted_text_{page_i}_{page['source']}",
                disabled=True,
            )

    with st.expander("🧩 View every indexed chunk", expanded=True):
        for chunk_i, chunk in enumerate(chunks, start=1):
            chunk_label = chunk["source"]
            if chunk["page"] is not None:
                chunk_label += f" — Page {chunk['page']}"
            st.markdown(f"**Chunk {chunk_i} · {chunk_label}**")
            st.caption(
                f'{len(chunk["text"].split())} words · {len(chunk["text"])} characters'
            )
            st.write(chunk["text"])
            st.divider()

    st.divider()
    st.subheader("🧪 RAG Metrics & Evaluation")
    st.write(
        "Test whether hybrid search retrieves passages matching the expected "
        "source and/or keywords that you label for each question."
    )
    st.info(
        "These metrics measure retrieval quality against your labels. They do "
        "not by themselves prove that a generated answer is correct or faithful."
    )
    eval_json = st.text_area(
        "Test cases (JSON)",
        value=DEFAULT_EVAL_JSON,
        height=250,
        help="Each test case needs a question and expected_source and/or expected_keywords.",
    )
    eval_top_k = st.slider(
        "Evaluate top K passages", min_value=1, max_value=10, value=TOP_K
    )
    if st.button("▶️ Run retrieval evaluation", type="primary", use_container_width=True):
        try:
            test_cases = json.loads(eval_json)
            if not isinstance(test_cases, list) or not test_cases:
                raise ValueError("JSON must be a non-empty list of test-case objects.")
            if any(not isinstance(item, dict) for item in test_cases):
                raise ValueError("Every test case must be a JSON object.")
            if not any(
                item.get("expected_source") or item.get("expected_keywords")
                for item in test_cases
            ):
                raise ValueError(
                    "Add expected_source and/or expected_keywords to your test cases."
                )

            with st.spinner("Running retrieval tests..."):
                eval_rows = evaluate_retrieval(
                    test_cases, chunks, index, embedding_model, top_k=eval_top_k
                )

            if not eval_rows:
                st.warning("No valid questions were found in the test set.")
            else:
                hit_rate = sum(row["hit"] for row in eval_rows) / len(eval_rows)
                mrr = sum(row["reciprocal_rank"] for row in eval_rows) / len(eval_rows)
                mean_precision = sum(
                    row["precision_at_k"] for row in eval_rows
                ) / len(eval_rows)

                metric_cols = st.columns(3)
                metric_cols[0].metric(
                    "Hit rate", f"{hit_rate:.0%}",
                    help="Share of questions with at least one matching passage in top K.",
                )
                metric_cols[1].metric(
                    "MRR", f"{mrr:.3f}",
                    help="Mean reciprocal rank of the first matching passage; higher is better.",
                )
                metric_cols[2].metric(
                    "Mean precision@K", f"{mean_precision:.3f}",
                    help="Matching passages divided by retrieved passages, averaged across questions.",
                )

                st.dataframe(
                    [{
                        "Question": row["question"],
                        "Hit": "✅" if row["hit"] else "❌",
                        "Reciprocal rank": round(row["reciprocal_rank"], 3),
                        "Precision@K": round(row["precision_at_k"], 3),
                        "Top sources": row["top_sources"],
                    } for row in eval_rows],
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("Inspect retrieved passages for each test"):
                    for row in eval_rows:
                        status = "✅" if row["hit"] else "❌"
                        st.markdown(f"**{status} {row['question']}**")
                        st.caption(
                            f"Expected source: {row['expected_source'] or 'not specified'} · "
                            f"Expected keywords: {row['expected_keywords'] or 'not specified'}"
                        )
                        st.write(row["top_passage_preview"])
                        st.divider()

                st.download_button(
                    "⬇️ Download evaluation results (JSON)",
                    data=json.dumps(eval_rows, indent=2, ensure_ascii=False),
                    file_name="rag_retrieval_evaluation.json",
                    mime="application/json",
                    use_container_width=True,
                )
        except Exception as error:
            st.error(f"Could not run evaluation: {error}")

    st.markdown("**How to read the metrics**")
    st.markdown("""- **Hit rate:** fraction of questions with at least one matching passage in top K.
- **MRR:** rewards relevant passages appearing near the top.
- **Precision@K:** fraction of retrieved passages that match your labels.

Use 10–30 representative questions for a more meaningful estimate. Include both answerable questions and questions whose answers are absent from the documents.""")
    st.stop()



# --------------------------------------------------
# 12. INDEXED CHUNK PREVIEW
# --------------------------------------------------
with st.expander("🔍 Preview indexed chunks"):
    for i, chunk in enumerate(chunks, start=1):
        location = chunk["source"]
        if chunk["page"] is not None:
            location += f" — Page {chunk['page']}"
        st.markdown(f"**Chunk {i} · {location}**")
        st.write(chunk["text"])
        st.divider()


# --------------------------------------------------
# 13. CHAT INTERFACE
# --------------------------------------------------
st.subheader("💬 Chat with your documents")
st.caption("Open a source panel to verify retrieved passages and citation labels against the uploaded text.")

button_col, _ = st.columns([1, 5])
with button_col:
    st.button(
        "🧹 Clear chat",
        on_click=clear_chat,
        use_container_width=True,
    )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Sources used"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['label']} — {source['location']}**"
                    )
                    st.caption(
                        "Semantic: "
                        f"{source['similarity']:.3f} · "
                        "BM25: "
                        f"{source['bm25_score']:.3f} · "
                        "Hybrid: "
                        f"{source['hybrid_score']:.3f}"
                    )
                    st.write(source["text"])

question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    # Preserve the earlier history for rewriting and answer generation.
    previous_history = st.session_state.chat_history.copy()
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found in your .env file."
            )

        client = load_gemini_client(api_key)

        with st.spinner("Understanding your question..."):
            search_query = rewrite_query(
                client,
                question,
                previous_history,
            )

        with st.expander("🔎 Search details"):
            st.write(f"**Original question:** {question}")
            st.write(f"**Query used for retrieval:** {search_query}")
            st.caption(
                f"Semantic threshold: {MIN_SEMANTIC_SCORE:.2f} · "
                f"Semantic weight: {SEMANTIC_WEIGHT:.0%} · "
                f"BM25 weight: {BM25_WEIGHT:.0%}"
            )

        with st.spinner("Searching your documents..."):
            retrieved_chunks = retrieve_chunks(
                search_query,
                chunks,
                index,
                embedding_model,
                top_k=TOP_K,
            )

        if not retrieved_chunks:
            answer = (
                "I couldn't find sufficiently relevant passages in the "
                "uploaded documents to answer that question. Try using "
                "different wording or checking whether the information "
                "is present in the uploaded files."
            )
            source_details = []
        else:
            with st.spinner("Generating an answer with Gemini..."):
                answer = generate_answer(
                    client,
                    question,
                    retrieved_chunks,
                    previous_history,
                )

            source_details = []
            for i, chunk in enumerate(retrieved_chunks, start=1):
                location = chunk["source"]
                if chunk["page"] is not None:
                    location += f" — Page {chunk['page']}"

                source_details.append({
                    "label": f"[S{i}]",
                    "location": location,
                    "text": chunk["text"],
                    "similarity": chunk["similarity"],
                    "bm25_score": chunk["bm25_score"],
                    "hybrid_score": chunk["hybrid_score"],
                })

        with st.chat_message("assistant"):
            st.markdown(answer)

            if source_details:
                with st.expander("📚 Retrieved source passages"):
                    for source in source_details:
                        st.markdown(
                            f"**{source['label']} — {source['location']}**"
                        )
                        st.caption(
                            f"Semantic similarity: {source['similarity']:.3f} · "
                            f"BM25: {source['bm25_score']:.3f} · "
                            f"Hybrid score: {source['hybrid_score']:.3f}"
                        )
                        st.write(source["text"])
                        st.divider()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": source_details,
        })

    except Exception as error:
        error_message = (
            "I couldn't generate an answer. Check your API key, model name, "
            "quota, and internet connection."
        )

        with st.chat_message("assistant"):
            st.error(error_message)
            st.caption(f"Technical details: {error}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": error_message,
        })
