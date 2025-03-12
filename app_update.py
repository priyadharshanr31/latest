import streamlit as st
from utils.file_loader import extract_text
from utils.retriever import create_temp_db, add_to_db, search_similar
from transformers import BartTokenizer, BartForConditionalGeneration

# 🚀 Load your trained student model
@st.cache_resource
def load_student_model():
    model_path = "C:\\CS Tech\\Agents\\Daily\\NEW\\updated_bart_student_model_v2"
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer

# 💬 Generate answer using student model
def generate_answer(model, tokenizer, context, question):
    full_prompt = f"""
You are a strict insurance assistant. Only use the context below to answer the user's question.

Instructions:
- Do NOT guess or add information.
- Answer in one sentence only.
- If answer not found, reply: "Not available in context."

Context:
{context}

Question: {question}
Answer:
"""

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=32,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        repetition_penalty=2.5,
        do_sample=False,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().replace("\n", " ")

# 🎯 Streamlit UI setup
st.set_page_config(page_title="Insurance Chatbot")
st.title("🛡️ Insurance Assistant Chatbot")

# Load model
model, tokenizer = load_student_model()

# File upload & user question
uploaded_file = st.file_uploader("📄 Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
query = st.text_input("💬 Ask your insurance question")

# Optional debug checkbox
show_debug = st.checkbox("🔍 Show Prompt & Retrieved Context", value=False)

# Main app logic
if uploaded_file and query:
    text = extract_text(uploaded_file)

    if not text.strip():
        st.warning("⚠️ Could not extract text. Please try another file.")
    else:
        # Split into 500-character chunks
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]

        # ChromaDB setup
        collection = create_temp_db()
        add_to_db(collection, chunks)

        # Search relevant chunks
        relevant_chunks = search_similar(collection, query)
        context = " ".join(relevant_chunks)

        # Debug info
        if show_debug:
            st.markdown("### 📄 Retrieved Context")
            st.info(context)

        # Generate and show answer
        answer = generate_answer(model, tokenizer, context, query)
        st.markdown("### 🤖 Answer:")
        st.success(answer)
