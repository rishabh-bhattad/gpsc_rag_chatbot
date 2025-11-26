import os
import fitz
import re
import docx
from dateutil import parser
from db_utils import get_chroma_collection


def extract_date(text: str):
    month_regex = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+"
    date_regex = r"\d{1,2}(?:st|nd|rd|th)?(?:,\s*|\s+)"
    year_regex = r"\d{4}"
    month_first_pattern = month_regex + date_regex + year_regex
    day_first_pattern = date_regex + month_regex + year_regex
    result = re.search(month_first_pattern, text, re.IGNORECASE)
    if result is None:
        result = re.search(day_first_pattern, text, re.IGNORECASE)
    if result:
        return result.group(0)
    else:
        return None


def extract_date_from_filename_fallback(filename: str):
    match = re.search(r"\d{1,2}[-_]\d{1,2}[-_]\d{2,4}", filename)
    if match:
        return normalize_date(match.group(0).replace("_", "-"))
    return None


def normalize_date(date: str = None):
    if date:
        date_cleaned = re.sub(r"(\d)(st|nd|th|rd)", r"\1", date)
        try:
            dt_object = parser.parse(date_cleaned)
            return dt_object.strftime("%Y-%m-%d")
        except:
            return None
    else:
        return None


def parse_pdf(filepath: str):
    doc = fitz.open(filename=filepath)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc_text = "\n".join(pages)
    date = normalize_date(extract_date(doc_text))
    return doc_text, date


def parse_docx(filepath: str):
    doc = docx.Document(filepath)
    paragraphs = []
    for para in doc.paragraphs:
        paragraphs.append(para.text)
    doc_text = "\n".join(paragraphs)
    date = normalize_date(extract_date(doc_text))
    return doc_text, date


def chunk_text(text: str, date: str, chunk_size: int = 2000, overlap:int = 400):
    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_content = text[start : end]
        chunks.append(
            {
                "text": chunk_content,
                "date": date
            }
        )

        start += (chunk_size - overlap)
        if end == text_length:
            break

    return chunks


def main():
    chroma_collection = get_chroma_collection("gpsc_pilot_v1")
    for dirpath, dirnames, filenames in os.walk("data/raw"):
        for filename in filenames:
            if filename.endswith(".pdf"):
                file_text, date = parse_pdf(os.path.join(dirpath, filename))
            elif filename.endswith(".docx"):
                file_text, date = parse_docx(os.path.join(dirpath, filename))
            else:
                continue
            date = date  if date else extract_date_from_filename_fallback(filename=filename)
            safe_date = date if date else "Unknown"
            chunks = chunk_text(text=file_text, date=date)
            ids, documents, metadata = [], [], []
            for index, chunk in enumerate(chunks):
                unique_id = f"{filename}_chunk_{index}"
                document = chunk['text']
                ids.append(unique_id)
                documents.append(document)
                metadata.append({"source": filename, "date": safe_date})
            if ids:
                chroma_collection.upsert(ids=ids, documents=documents, metadatas=metadata)
                print(f"Saved {len(ids)} chunks from {filename}.")
            else:
                print(f"No text found in {filename}.")

if __name__ == "__main__":
    main()
