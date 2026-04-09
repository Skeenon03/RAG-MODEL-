import sys
import os
import json
import re 
import pandas as pd
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOCS_PATH = os.path.join(BASE_DIR, "docs")
DB_PATH = os.path.join(BASE_DIR, "db", "faiss_index")
LOG_FILE = "processed_files.json"


TESSERACT_EXE_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_BIN_PATH = r'C:\Program Files\Release-25.12.0-0\poppler-25.12.0\Library\bin'


if TESSERACT_EXE_PATH:
   pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE_PATH

# HELPER FUNCTIONS 

def get_processed_files():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_processed_files(processed_files):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(list(processed_files), f, indent=4)

def clean_text(text):
   

    if not text: return ""

    # 1. Remove common OCR noise characters
    
    text = re.sub(r'[|_]', ' ', text)
    
    # 2. Remove long sequences of dashes or dots 
    text = re.sub(r'[-.]{3,}', '', text)

    # 3. Collapse multiple spaces and newlines into single ones
    text = re.sub(r'\s+', ' ', text).strip()

    # 4. Remove "garbage" lines that are too short to be real text

    if len(text) < 3:
        return ""

    return text

def ocr_pdf_page(file_path, page_number):
    try:
        images = convert_from_path(
            file_path, 
            first_page=page_number, 
            last_page=page_number,
            poppler_path=POPPLER_BIN_PATH 
        )
        if images:
            # Raw extraction
            raw_text = pytesseract.image_to_string(images[0])
            
            # CLEANING STEP

            return clean_text(raw_text)
            
    except Exception as e:
        print(f"      ⚠️ OCR Failed for page {page_number}: {e}")
    return ""

def load_documents():
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"The directory {DOCS_PATH} does not exist.")
    
    processed_files = get_processed_files()
    documents = []
    new_files_found = []

    print(f"Checking for new files in '{DOCS_PATH}'...")

    for root, dirs, files in os.walk(DOCS_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            file_path = os.path.normpath(file_path)
            
            if file_path not in processed_files:
                try:
                    docs = [] 
                    
                    # --- 1. Text Files ---
                    if file.lower().endswith(".txt"):
                        loader = TextLoader(file_path, encoding="utf-8")
                        raw_docs = loader.load()
                        # Clean text files too just in case
                        for d in raw_docs:
                            d.page_content = clean_text(d.page_content)
                            if d.page_content: docs.append(d)

                    # 2. PDFs -
                    elif file.lower().endswith(".pdf"):
                        print(f"   -> Processing PDF: {file}")
                        
                        with pdfplumber.open(file_path) as pdf:
                            for i, page in enumerate(pdf.pages):
                                page_num = i + 1
                                
                                
                                text = page.extract_text()
                                
                                # B. Check if Scanned
                                if not text or len(text.strip()) < 50:
                                    print(f"      📷 Page {page_num} looks scanned. Running OCR...")
                                    text = ocr_pdf_page(file_path, page_num)
                                    source_tag = "Scanned Data"
                                else:
                                    
                                    text = clean_text(text)
                                    source_tag = "Context"

                                if text:
                                    new_doc = Document(
                                        page_content=f"[Source: {file} | Page: {page_num} | {source_tag}]\n{text}",
                                        metadata={"source": file_path, "page": page_num, "type": "text"}
                                    )
                                    docs.append(new_doc)
                        
                    #  3. Excel Files 
                    elif file.lower().endswith(".xlsx"):
                        print(f"   -> Processing Excel: {file}")
                        df = pd.read_excel(file_path)
                        df = df.fillna("None") 
                        
                        for index, row in df.iterrows():
                           
                            row_text = "\n".join([f"{col}: {val}" for col, val in row.items()])
                            new_doc = Document(
                                page_content=f"[Source: {file} | Row: {index+1} | Excel Data]\n{row_text}",
                                metadata={"source": file_path, "page": index + 1, "type": "excel_row"}
                            )
                            docs.append(new_doc)

                    if docs:
                        documents.extend(docs)
                        new_files_found.append(file_path)
                        if not file.lower().endswith(".xlsx") and not file.lower().endswith(".pdf"): 
                            print(f"   -> Found new file: {file}")
                    
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
            
    if not documents:
        print("No new files found.")
        return [], []

    print(f"Loaded {len(documents)} new documents.")
    return documents, new_files_found

def split_documents(documents):
    if not documents: return []
    print("Splitting documents...")
    
    docs_to_split = [d for d in documents if d.metadata.get('type') == 'text']
    docs_already_split = [d for d in documents if d.metadata.get('type') in ['excel_row']]

    chunks = []
    if docs_to_split:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs_to_split)
    
    return chunks + docs_already_split

def update_vector_store(chunks):
    if not chunks: return
    
    print("Loading embedding model (gte-large)...")
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    
    if os.path.exists(DB_PATH):
        print("Loading existing database...")
        vectorstore = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("Creating new database...")
        vectorstore = FAISS.from_documents([chunks[0]], embedding_model)
        chunks = chunks[1:] 

    batch_size = 32
    print(f"Embedding {len(chunks)} chunks...")
    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing"):
        vectorstore.add_documents(chunks[i:i + batch_size])
        if i % (batch_size * 5) == 0: vectorstore.save_local(DB_PATH)

    vectorstore.save_local(DB_PATH)
    print(f"✅ Database saved.")

def main():
    documents, new_files = load_documents()
    if documents:
        chunks = split_documents(documents)
        update_vector_store(chunks)
        processed = get_processed_files()
        processed.update(new_files)
        save_processed_files(processed)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()