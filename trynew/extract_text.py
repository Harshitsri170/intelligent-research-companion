# extract_text.py
from PyPDF2 import PdfReader

def extract_text_from_pdf(uploaded_file):
    try:
        raw_text = ''
        pdfreader = PdfReader(uploaded_file)
        for page_num, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                raw_text += content
        if not raw_text:
            raise ValueError("No text extracted from PDF.")
        return raw_text
    except Exception as e:
        print(f"[ERROR] extract_text_from_pdf: {e}")
        raise