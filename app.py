import streamlit as st
import fitz
import pdfplumber
from PIL import Image
import io, base64, os
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import chromadb
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# ---------------- Load API key ----------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Gemini API key not found in .env")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Models ----------------
text_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---------------- Chroma ----------------
chroma_client = chromadb.Client()
text_collection = chroma_client.get_or_create_collection("multimodal_texts")  # 384-dim
image_collection = chroma_client.get_or_create_collection("multimodal_images")  # 512-dim

# ---------------- Helpers ----------------
def embed_text(text: str):
    return text_model.encode(text).tolist()

def embed_image(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy().tolist()

def pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def decode_base64_image(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

# ---------------- PDF ingestion ----------------
def ingest_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    # Text
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text and text.strip():
            text_collection.add(
                documents=[text],
                embeddings=[embed_text(text)],
                metadatas=[{"type": "text", "page": page_num, "source_pdf":pdf_file.name}],
                ids=[f"text_{pdf_file.name}_{page_num}"]
            )

    #  Tables
    pdf_file.seek(0)
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()
            for i, table in enumerate(tables):
                table_text = "\n".join([
                    "\t".join([str(cell) if cell is not None else "" for cell in row])
                    for row in table if row
                ])
                if table_text.strip():
                    text_collection.add(
                        documents=[table_text],
                        embeddings=[embed_text(table_text)],
                        metadatas=[{"type": "table", "page": page_num, "table_index": i}],
                        ids=[f"table_{page_num}_{i}"]
                    )

    # Images
    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            image_collection.add(
                documents=["[IMAGE]"],
                embeddings=[embed_image(image)],
                metadatas=[{
                    "type": "image",
                    "page": page_num,
                    "image_b64": pil_to_base64(image)
                }],
                ids=[f"image_{page_num}_{img_index}"]
            )

# ---------------- Retrieval ----------------
def retrieve_text(query, n_results=50):
    query_emb = embed_text(query)
    results = text_collection.query(query_embeddings=[query_emb], n_results=n_results)
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })
    return retrieved

def retrieve_images(query_emb, n_results=50):
    results = image_collection.query(query_embeddings=[query_emb], n_results=n_results)
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "metadata": results["metadatas"][0][i],
        })
    return retrieved

# ---------------- QA chain ----------------
QA_PROMPT = """Answer the question using the provided context.
If the answer is not available, say 'answer is not available in the context'.
Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""

def get_qa_chain(template):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, api_key=api_key)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ---------------- Streamlit ----------------
def main():
    st.set_page_config(page_title="Document QA AI Agent")
    st.header("📑 Document QA AI Agent")

    with st.sidebar:
        st.title("Upload PDFs")
        pdfs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdfs:
                st.warning("Upload at least one PDF")
            else:
                with st.spinner("Processing PDFs..."):
                    for pdf in pdfs:
                        ingest_pdf(pdf)
                    st.success("PDFs are uploaded successfully!")

    query = st.text_input("Ask a question")
    if query:
        # Retrieve text
        text_results = retrieve_text(query, n_results=3)
        context_texts = [item["document"] for item in text_results if item["document"] and item["document"].strip()]

        # Display images (optional)
        query_emb = embed_text(query)
        img_results = retrieve_images(query_emb, n_results=3)
        for item in img_results:
            img = decode_base64_image(item["metadata"]["image_b64"])
            st.image(img, caption=f"Retrieved image (page {item['metadata']['page']})")

        if context_texts:
            context_documents = [Document(page_content=text) for text in context_texts]
            chain = get_qa_chain(QA_PROMPT)
            resp = chain({"input_documents": context_documents, "question": query}, return_only_outputs=True)
            st.write("💡 Answer:", resp["output_text"])
        else:
            st.write("💡 No text context available to answer this question.")

if __name__ == "__main__":
    main()
