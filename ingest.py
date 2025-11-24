import os
import fitz
import re
import docx
from dateutil import parser


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


def chunk_text(text: str, date: str, chunk_size: int = 1000, overlap:int = 200):
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


text, date = parse_pdf("data/raw/GC Minutes_10-31-18.pdf")
print(date)
text, date = parse_docx("data/raw/GPSC GC Meeting Minutes 2_13_2025.docx")
print(date)

print(chunk_text(text, date))
