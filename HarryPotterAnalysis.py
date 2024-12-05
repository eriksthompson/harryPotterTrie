# importing required modules
from pypdf import PdfReader
import re
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

# creating a pdf reader object
reader = PdfReader('Sorcerer\'sStone.pdf')

# printing number of pages in pdf file
#print(len(reader.pages))

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

with open('text_analysis.txt', 'w') as file:
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