import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import os
import nltk
from nltk.tokenize import word_tokenize

# ── NLTK Downloads ─────────────────────────────────────────────────
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# ── Model Definition ───────────────────────────────────────────────
class BiLSTMQuora(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout=0.3):
        super(BiLSTMQuora, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        emb = self.dropout(self.embedding(x))
        _, (hn, _) = self.bilstm(emb)
        return torch.cat((hn[-2], hn[-1]), dim=1)

    def forward(self, q1, q2):
        q1_enc = self.encode(q1)
        q2_enc = self.encode(q2)
        combined = torch.cat((q1_enc, q2_enc), dim=1)
        return self.classifier(combined).squeeze(1)


# ── Helper Functions ───────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def pad_or_truncate(ids, max_len):
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - len(ids))


def predict(question1, question2, model, vocab, device,
            max_len_q1=128, max_len_q2=249):
    model.eval()

    q1_tokens = word_tokenize(clean_text(question1))
    q2_tokens = word_tokenize(clean_text(question2))

    q1_ids = pad_or_truncate([vocab.get(t, 1) for t in q1_tokens], max_len_q1)
    q2_ids = pad_or_truncate([vocab.get(t, 1) for t in q2_tokens], max_len_q2)

    q1_t = torch.tensor([q1_ids], dtype=torch.long).to(device)
    q2_t = torch.tensor([q2_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        prob = model(q1_t, q2_t).item()

    return prob


# ── Load Model ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cpu")  # Hugging Face free tier is CPU only

    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    with open("config.pkl", "rb") as f:
        config = pickle.load(f)

    model = BiLSTMQuora(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    model.load_state_dict(torch.load(
        "bilstm_quora.pt",
        map_location=device
    ))
    model.eval()

    return model, vocab, config, device


# ── Streamlit UI ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Quora Duplicate Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Quora Duplicate Question Detector")
st.markdown("Enter two questions to check if they are duplicates.")
st.divider()

with st.spinner("Loading model..."):
    model, vocab, config, device = load_model()

q1 = st.text_area(
    label="Question 1",
    placeholder="e.g. How do I improve my English?",
    height=100
)

q2 = st.text_area(
    label="Question 2",
    placeholder="e.g. What are the ways to get better at English?",
    height=100
)

st.divider()

if st.button("Check", use_container_width=True, type="primary"):
    if not q1.strip() or not q2.strip():
        st.warning("⚠️ Please enter both questions.")
    else:
        with st.spinner("Analyzing..."):
            prob = predict(q1, q2, model, vocab, device,
                           max_len_q1=config['max_len_q1'],
                           max_len_q2=config['max_len_q2'])

        st.subheader("Result")

        if prob >= 0.5:
            st.success("✅ These questions are **Duplicate**")
        else:
            st.error("❌ These questions are **Not Duplicate**")

        st.markdown(f"**Confidence:** `{prob:.2%}`")
        st.progress(float(prob))

        with st.expander("See details"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Duplicate Probability", f"{prob:.4f}")
            with col2:
                st.metric("Not Duplicate Probability", f"{1 - prob:.4f}")