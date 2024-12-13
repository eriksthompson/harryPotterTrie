from pypdf import PdfReader
import sys
import re
bookName = input('Enter file of text to analyze (Empty for Harry Potter):')
def is_valid_pdf_filename(filename):
    # Define a regex pattern for invalid filename characters
    invalid_chars_pattern = r'[<>:"/\\|?*]'
    
    # Check if the filename contains invalid characters or doesn't end with ".pdf"
    if re.search(invalid_chars_pattern, filename) or not filename.lower().endswith('.pdf'):
        return False
    return True

# creating a pdf reader object
if bookName == '':
    bookName = 'Sorcerer\'sStone.pdf'

if not is_valid_pdf_filename(bookName):
    print("Invalid file. File must exist and end with .pdf")
    sys.exit()

reader = PdfReader(bookName)

print('Generating pages ', '1-', len(reader.pages))
#page = reader.pages[1]
#text = page.extract_text()
text = []
for i in range(0, len(reader.pages)):
    page = reader.pages[i]
    text.append(page.extract_text())

with open("book_text.txt", "w", encoding="utf-8") as file:
    for t in text:
        file.write(t)
        file.write('\n\n')

