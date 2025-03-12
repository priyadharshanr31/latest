import os
import docx
import PyPDF2

def read_txt(file):
    return file.read().decode("utf-8")

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text(file):
    ext = os.path.splitext(file.name)[-1].lower()
    if ext == ".txt":
        return read_txt(file)
    elif ext == ".docx":
        return read_docx(file)
    elif ext == ".pdf":
        return read_pdf(file)
    else:
        return ""
