BookAnalysis.py is the main script for the project.
book_report.txt is the summarization output of the AI model.
pdf_to_text.py is needed for first converting the pdf books into .txt files to input into BookAnalysis.py

Option 1 counts characters and words and does some basic analysis of average word repeated occurrences. 
It uses a Trie to find how many words are prefixes of a previous word.
Dictionary utilized to count occurrences of characters and words.

Option 2 allows for AI modeling and generation of a book report summary. Each page is summarized and combined.
Model and tokenizer are t5-small from Hugging Face. Cuda uses GPU hardware to increase performance. 
Pipeline function is a high level abstraction to increase I/O flow. num_beams parameter is 3 which is lower than average 5-8. This parameter controls beam search and increases quality of output text with higher settings.

Run time: 3.05 minutes on a 250 page book (Harry Potter and the Sorcerer's Stone). This is reasonable since each page is about 300-500 tokens and was computed in batches of 16 pages at a time. Previously run time was 45 minutes without GPU cuda utilization or pipeline abstraction.


Dependencies: pypdf, transformers, sentencepiece, torch, tensorflow,and flax, tf-keras from pip. numba and cudatoolkit from conda.
