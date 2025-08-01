import fitz

def extract_text_from_pdfs(uploaded_files):
    all_text = []
    for file in uploaded_files:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        full_text = "\n".join([page.get_text() for page in doc])
        all_text.append(full_text)
    return all_text
