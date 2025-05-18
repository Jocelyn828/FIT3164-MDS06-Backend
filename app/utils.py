from django.db import connection
import tempfile
import os
import docx
import PyPDF2

def reset_sequence(model):
    # Used to reset the primary key sequence of a model (id field) to 1
    # when deleting all objects from the table.
    table_name = model._meta.db_table
    if connection.vendor == 'postgresql':
        sql = f"ALTER SEQUENCE {table_name}_id_seq RESTART WITH 1"
    elif connection.vendor == 'sqlite':
        sql = f"DELETE FROM sqlite_sequence WHERE name='{table_name}'"
    elif connection.vendor == 'mysql':
        sql = f"ALTER TABLE {table_name} AUTO_INCREMENT = 1"
    else:
        return  # Unsupported DB backend
    
    with connection.cursor() as cursor:
        cursor.execute(sql)


def extract_text_from_document(document):
    """Extract text from various document formats with debugging"""
    file_extension = os.path.splitext(document.name)[1].lower()
   
    # Create a temp file to work with
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        for chunk in document.chunks():
            temp.write(chunk)
   
    try:
        text = ""
       
        # Process based on file type
        if file_extension == '.txt':
            with open(temp.name, 'r', encoding='utf-8') as f:
                text = f.read()
               
        elif file_extension == '.docx':
            doc = docx.Document(temp.name)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
           
        elif file_extension == '.pdf':
            with open(temp.name, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        
        # Log the extraction results
        print(f"Extracted {len(text)} characters from {document.name}")
        if len(text.strip()) < 50:
            print(f"WARNING: Very little text extracted from {document.name}")
               
        return text
   
    finally:
        # Clean up the temp file
        os.unlink(temp.name)