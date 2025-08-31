import requests
import PyPDF2
from io import BytesIO
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class EasyRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.api_key = None

    def set_api_key(self, key):
        self.api_key = key

    def extract_text(self, file):
        """Extract text from PDF or TXT safely"""
        try:
            if file.type == "application/pdf":
                pdf = PyPDF2.PdfReader(BytesIO(file.read()))
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                return text.strip()
            elif file.type == "text/plain":
                return file.read().decode("utf-8").strip()
        except:
            return ""
        return ""

    def process_documents(self, files):
        """Process uploaded files and create vector store"""
        texts = [self.extract_text(f) for f in files]
        texts = [t for t in texts if t]  # remove empty texts
        if not texts:
            return False, "No valid text found in uploaded files"

        docs = [Document(page_content=t) for t in texts]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        if not chunks:
            return False, "Text chunks are empty after splitting"

        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return True, f"Processed {len(files)} files into {len(chunks)} chunks"

    def search_documents(self, question):
        if not self.vector_store:
            return ""
        docs = self.vector_store.similarity_search(question, k=2)
        return "\n".join([d.page_content for d in docs])[:800]

    def generate_answer(self, question, context):
        if not self.api_key:
            return "API key not set"

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        try:
            res = requests.post(
                "https://api-inference.huggingface.co/models/google/flan-t5-base",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": prompt, "parameters": {"max_length": 100}},
                timeout=30
            )
            if res.status_code == 200:
                result = res.json()
                answer = result[0].get("generated_text", "").replace(prompt, "").strip()
                return answer if answer else "No answer found"
        except:
            pass
        # Fallback: return first sentence from context
        return context.split(".")[0] + "."

    def ask_question(self, question):
        if not self.vector_store:
            return "Please upload documents first"
        context = self.search_documents(question)
        return self.generate_answer(question, context)
