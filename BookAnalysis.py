#Dependencies: pypdf, transformers, sentencepiece, torch, tensorflow,and flax from pip. numba and cudatoolkit from conda.
#First time running the program in 2-book report mode it will take longer to create AI model.
# importing required modules
import torch
from pypdf import PdfReader
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word_count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
    # Insert a word into the trie
    def insert(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True
        current.word_count += 1
    # Search for a word in the trie
    def search(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                return False
            current = current.children[char]
        return current.is_end_of_word

    # Check if any word starts with a given prefix
    def starts_with(self, prefix):
        current = self.root
        for char in prefix:
            if char not in current.children:
                return False
            current = current.children[char]
        return True
    def count_words(self):
        def dfs(node):
            # Base count: 1 if the current node ends a word
            count = 1 if node.is_end_of_word else 0
            # Recursively count words in all children
            for child in node.children.values():
                count += dfs(child)
            return count

        # Start DFS from the root
        return dfs(self.root)

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

# printing number of pages in pdf file
#print(len(reader.pages))

analysisType = input('Enter 1 for character and word counts or 2 for AI text to text book report: ')
while (analysisType != '1' and analysisType != '2'):
    analysisType = input('Try again. Enter 1 for character and word counts or 2 for AI text to text book report: ')

if analysisType == '1':
    #Analysis Type: character and word analysis using dict and Trie.
    wordTrie = Trie()
    letter_count = {}
    #Outer loop through pages of pdf book.
    total_words = 0
    total_prefix = 0
    total_is_prefix = 0
    word_count2 = {}
    for i in range(1, len(reader.pages)):
        #Getting specific page from pdf file
        page = reader.pages[i]
        text = page.extract_text()
        text = text.lower()
        #Take out all non alpha characters
        text = ''.join([i for i in text if i.isalpha() or i == ' ' or i == '\n'])

        words = text.split()
        for w in words:
            
            #Count letters
            for c in w:
                if c not in letter_count.keys():
                    letter_count[c] = 1
                else:
                    letter_count[c] = letter_count[c] + 1

            #Check if word is prefix in wordTrie
            if wordTrie.starts_with(w):
                total_is_prefix += 1
            
            #Check all prefixes of current is in word trie
            for i2 in range(len(w)):
                if wordTrie.search(w[:i2]):
                    total_prefix +=1
                    break
            wordTrie.insert(w)
            if w not in word_count2:
                word_count2[w] = 1
            else:
                word_count2[w] = word_count2[w] + 1
        total_words += len(words)
    #Data analysis
    #Count all unique words in wordTrie
    def count_unique(node):
        count = 0
        end_word = 0
        if node.is_end_of_word:
            end_word += 1
        for c in node.children.keys():
            end_word += count_unique(node.children[c])
        return end_word

    def populate_word_count(node, dict1, current=''):
        for c in node.children.keys():
            current1 = current + c
            if node.is_end_of_word:
                dict1[current1] = node.word_count
            populate_word_count(node.children[c], dict1, current1)

    word_count = {}

    populate_word_count(wordTrie.root, word_count)



    with open('word_character_analysis.txt', 'w') as file:
        file.write('Unique number of words in Harry Potter Sorcerer\'s Stone: '+ str(wordTrie.count_words())+ '\n')
        file.write('Total number of words in Harry Potter Sorcerer\'s Stone: ' + str(total_words)+ '\n')
        file.write('Average number of times words are repeated in book: ' + str(total_words/wordTrie.count_words())+ '\n')
        # Loop through the dictionary in descending order by integer values
        file.write('Count of occurrences of each letter in alphabet:'+ '\n')
        for char, value in sorted(letter_count.items(), key=lambda item: item[1], reverse=True):
            file.write(f"Character: {char}, Value: {value}"+ '\n')
        file.write('Number of words with a previous word as a prefix: ' + str(total_prefix)+ '\n')
        file.write('Number of words with current word as prefix of previous word: ' + str(total_is_prefix) + '\n')
        file.write('Word Count in descending order of occurrences:' + ' \n')


        #Attempted finding word occurrences with Trie and failed.
        #Found word occurrences with Dictionary.
        for word, value in sorted(word_count2.items(), key=lambda item: item[1], reverse=True)[:15]:
            file.write(f"Word: {word}, Count: {value}"+ '\n')


# getting a specific page from the pdf file
#page = reader.pages[1]

# extracting text from page
#text = page.extract_text()
#print(text)


if analysisType == '2':
    print('GPU Dependency- Is torch CUDA available? ' + 'Yes.' if torch.cuda.is_available() else 'No.')
    #If no install CUDA pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Load model and tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Move the model to GPU if available
    model = model.to(device)

    book_report = []
    for i in range(1, len(reader.pages), 100):

        #page = reader.pages[1]
        #text = page.extract_text()
        text = []
        for i2 in range(1,min(len(reader.pages),i+100)):
            text.append('Summarize: '+ reader.pages[i2].extract_text())
        #paragraphs = text.split('\n\n')
        #print(paragraphs)
        #for i in range(len(paragraphs)):
        # Tokenize input
        inputs = tokenizer.batch_encode_plus(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        # Extract tensors
        #print(inputs)
        input_ids = inputs["input_ids"]  # Tensor of token IDs
        attention_mask = inputs["attention_mask"]  # Tensor for attention mask
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        # Generate summary
        outputs = model.generate(input_ids=input_ids, 
                                    attention_mask=attention_mask,
                                    max_length=150,  # Set a higher max_length for longer summaries
                                min_length=50,   # Set a minimum length for the summary
                                num_beams=1,      # Use beam search for better generation quality
                                no_repeat_ngram_size=2,  # Avoid repeating n-grams (e.g., pairs of words)
                                repetition_penalty=5.0   # Penalize repetition
                                )
        # Decode each output
        summary1 = [tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True, errors='ignore') for output in outputs]
        book_report = book_report + summary1
    # Function to split text into smaller chunks
    def split_into_chunks(text, chunk_size=12):
        text = text.replace(',','')
        words = text.split(' ')
        text2 = ''
        for i in range(0, len(words), chunk_size):  # Iterate by chunk_size
            chunk = ' '.join(words[i:i+chunk_size])  # Join the words into a chunk
            text2 += chunk + '\n'  # Add the chunk to the final text with a newline
        return text2
    with open('book_report.txt', 'w') as file:    
        for i in range(len(book_report)):
            chunks = split_into_chunks(book_report[i])  # Split summary into chunks
            file.write(f'Page {i+1} Summary:\n')  # Write page header
            file.write(chunks)  # Write the chunks into the file
print('Execution complete')