<h1 id="content">NLP - Natural Language Processing</h1>
- <a href="#nlp-basics">Natural Language Processing (NLP) Basics</a>

<h1 id="nlp-basics">Natural Language Processing (NLP) Basics</h1>
<a href="#content">Back to Top</a>

### Natural Language Processing Bootcamp

- Set up Spacy and Language Library
- Understand Basic NLP Topics:
  - Tokenization
  - Stemming
  - Lemmatization
  - Stop Words
- Spacy for Vocabulary Matching

### What is Spacy

- For many NLP tasks, Spacy only has one implemented method, choosing the most efficient algorithm currently available (often do not have the option to choose other algorithms)

### What is NLTK

- Natural Langugage Toolkit (very popular open source)
- Initially released in 2001 (older than Spacy which released 2015)
- Provides many functionalities, but includes less efficient implementations

### NLTK vs Spacy

|     Requirements      |            NLTK             |                                             Spacy                                             |
| :-------------------: | :-------------------------: | :-------------------------------------------------------------------------------------------: |
| Many common NLP tasks |  Slower and less efficient  | Faster and more efficient (cost of user not being able to choose algorithmic implementations) |
|  Pre-created models   | Typically easier to perform |        Does not include pre-created models for some applications (sentiment analysis)         |

### What is NLP

- Natural Language Processing
- An area of computer science and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data
- Often when performing analysis, lots of data is numerical (sales numbers, physical measurements, quantifiable categories)
- Computers are very good at handling direct numerical information
- As humans, we can tell there is a plethora of information inside of text documents but a computer needs specialized processing techniques in order to "understand" raw text data
- Text data is highly unstructured and can be in multiple languages
- NLP attempts to use a variety of techniques in order to create structure out of text data (basic techniques - built into libraries - Spacy, NLTK)
- Example Use Cases:
  - Classifying Emails (Spam vs Legitimate)
  - Sentiment Analysis of Text Movie Reviews
  - Analyzing Trends from written customer feedback forms
  - Understanding text commands (Google - "Hey Google, play this song")

### NLP Basics

Refer to ![01-NLP-Python-Basics](01-NLP-Python-Basics)

1. ![Spacy Basics](01-NLP-Python-Basics/00-Spacy-Basics.ipynb)
   **nlp()** function from Spacy:

   - automatically takes raw text and performs a series of operations to tag, parse and describe the text data

   ![NLP Pipeline](images/pipeline.png)

2. ![Tokenization](01-NLP-Python-Basics/01-Tokenization.ipynb)

   - Notice that tokens are pieces of original text
   - Entity Recognition: we do not see any conversion to words stems or lemmas (base forms of words) and we have not seen anything about organizations/places/money
   - Tokens are the basic building blocks of a Doc object (everything that helps us understand the meaning of the text is derived from tokens and their relationship to one another)

   ![Tokenization](images/tokenization.png)

   - **Prefix**: Character(s) at the beginning `$ ( " ¿`
   - **Suffix**: Character(s) at the end `km ) , . ! "`
   - **Infix**: Character(s) in between `- -- / ...`
   - **Exception**: Special-case rule to split a string into several tokens or prevent a token from being split when punctuation rules are applied

3. ![Stemming](01-NLP-Python-Basics/02-Stemming.ipynb)

   - Often when searching text for a certain keyword, it helps if the search returns variations of the word (e.g. searching for "boat" might also return "boats" and "boating" - "boat" would be the stem for [boat, boater, boating, boats])
   - Stemming is somewhat crude method for cataloging related words
   - It essentially chops off letters from the end until the stem is reached
   - Works fairly well in most cases but unfortunately English has many exceptions where a more sophisticated process is required
   - Spacy does not include a stemmer, opting instead to rely entirely on lemmatization (use NLTK and learn about various Stemmers - Porter Stemmer and Snowball Stemmer)

   1. Porter's Algorithm

   - One of the most common and effective stemming tools
   - Developed by Martin Porter in 1980
   - The algorithm employs five phases of word reduction, each with its own set of mapping rules

   - First phase: simple suffix mapping rules are defined

     ![Stemming 1](images/stemming1.png)

     - From a given set of stemming rules only one rule is applied, based on the longest suffix S1 (e.g. caresses reduces to caress but not cares)

   - More sophisticated phases consider the length/complexity of the word before applying a rule

     ![Stemming 2](images/stemming2.png)

   2. Snowball

   - Name of a stemming language
   - Developed yb Martin Porter
   - The algorithm used here is more accurately called "English Stemmer" or "Porter2 Stemmer"
   - Offers a slight improvement over the original Porter stemmer (both in logic and speed)

   - First phase: simple suffix mapping rules are defined

     ![Stemming 1](images/stemming1.png)

     - From a given set of stemming rules only one rule is applied, based on the longest suffix S1 (e.g. caresses reduces to caress but not cares)

   - More sophisticated phases consider the length/complexity of the word before applying a rule

     ![Stemming 2](images/stemming2.png)

<h1 id="parts-of-speech">Parts of Speech Tagging</h1>
       <a href="#content">Back to Top</a>
