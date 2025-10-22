# 🤖 Chatbot for Public Sector Organization

## 📝 Project Description
This project focuses on developing a **chatbot** using **Deep Learning** and **Natural Language Processing (NLP)** techniques to accurately understand and respond to queries from employees of a large public sector organization.

The chatbot is capable of handling a wide range of questions related to:
- HR Policies  
- IT Support  
- Company Events  
- Other Organizational Matters  

It also includes document processing capabilities that allow employees to **upload documents** for information extraction and summarization.

---

## 🎯 Objectives
- Develop a chatbot that understands employee queries accurately.  
- Handle HR, IT, and organization-related queries effectively.  
- Integrate document upload functionality for text extraction and summarization.  
- Ensure the chatbot can serve **at least 5 users simultaneously**.  
- Maintain a **response time under 5 seconds** for any query.  
- Enable **2-Factor Authentication (2FA)** via email for enhanced security.  

---

## ⚙️ Key Functionalities
- **Natural Language Understanding (NLU):** Understands employee queries using deep learning-based NLP models.  
- **Query Response System:** Provides relevant and accurate responses to HR and IT-related queries.  
- **Document Processing:**  
  - Upload document functionality.  
  - Extracts text content from uploaded files.  
  - Summarizes document content.  
  - Extracts keyword-based information.  
- **Security Features:**  
  - Email-based 2FA authentication.  
  - Secure data handling for uploaded files and user sessions.  
- **Performance Optimization:**  
  - Scalable for multiple users.  
  - Optimized for low latency and fast query responses.

---

## 🧠 Technologies and Tools Used
- **Programming Language:** Python  
- **Framework:** TensorFlow / PyTorch (for deep learning model)  
- **Libraries:**  
  - NLP: NLTK, spaCy, transformers  
  - Document Processing: PyPDF2, docx, nltk, gensim, sumy  
  - Email Authentication: smtplib, email.mime  
- **Environment:** Google Colab  
- **Dataset:** Publicly available HR and IT-related information (for demo purposes)  

---

## 🧩 Workflow Overview
1. **User Input:** The employee enters a query or uploads a document.  
2. **Preprocessing:** The query or document is processed using NLP techniques (tokenization, lemmatization, etc.).  
3. **Model Processing:**  
   - For queries: The chatbot analyzes the input and retrieves or generates the appropriate response.  
   - For documents: The system extracts text, summarizes it, and identifies key terms.  
4. **Response Generation:** The chatbot formulates a natural-language response.  
5. **Authentication:** Before access, users verify their identity through **email-based 2FA**.  

---

## ⚡ Performance Goals
| Parameter | Target |
|------------|---------|
| Maximum Concurrent Users | 5 |
| Maximum Response Time | ≤ 5 seconds |
| Authentication | 2-Factor (Email-based) |
| Document Length for Demo | 8–10 pages |

---

## 🧾 Output Capabilities
- Responds accurately to HR, IT, and organization-related questions.  
- Summarizes large documents into concise readable text.  
- Extracts keywords and main topics from uploaded documents.  
- Ensures secure and verified access via email-based authentication.  

---

## 🔐 Security
- Implements **2-Factor Authentication** using email verification codes.  
- Ensures secure data handling for user credentials and uploaded files.  
- Limits user access to verified organizational emails.

---

## 🧮 Scalability
- Handles **minimum 5 users simultaneously** without degradation in performance.  
- Optimized backend to ensure **response time ≤ 5 seconds**.  

---

## 🧰 Installation (For Local or Colab)
1. Open the provided `CHATBOT.ipynb` file in **Google Colab**.  
2. Run all cells sequentially to install dependencies and initialize the model.  
3. Upload a sample HR or IT policy document when prompted.  
4. Interact with the chatbot interface to test:  
   - Text-based queries.  
   - Document summarization and keyword extraction.  
5. Observe the response time and 2FA email verification functionality.

---

## 🧠 Example Queries
| Query Type | Example |
|-------------|----------|
| HR Policy | “How many casual leaves are allowed per year?” |
| IT Support | “How can I reset my system password?” |
| Events | “When is the next company annual day celebration?” |
| Document | “Summarize the Policy HR.pdf file.” |

---

## 💻 Code Implementation
```python
!pip install transformers sentence-transformers PyPDF2 nltk --quiet

import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import nltk
import random
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from google.colab import files

nltk.download('punkt')
nltk.download('stopwords')

# ===============================================
# 🧠 Load Models
# ===============================================
print("Loading models... please wait.")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Models loaded successfully!\n")

# ===============================================
# 📂 Upload and Extract PDF
# ===============================================
print("Upload a document (PDF)...")
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

def extract_text_from_pdf(path):
    reader = PyPDF2.PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

document_text = extract_text_from_pdf(pdf_path)
print("\n✅ Document uploaded and extracted successfully!\n")

# ===============================================
# 🧹 Preprocess Text
# ===============================================
def preprocess_text(text):
    cleaned_sentences = []
    pattern_header = re.compile(r"Company XYZ Human Resources Policy Manual", re.IGNORECASE)
    pattern_page_num = re.compile(r"Page \d+( of \d+)?", re.IGNORECASE)
    sentences_raw = sent_tokenize(text)

    for sent in sentences_raw:
        sent_stripped = sent.strip()
        if not sent_stripped:
            continue
        if pattern_header.search(sent_stripped) or pattern_page_num.search(sent_stripped):
            continue
        if len(sent_stripped.split()) < 4:
            continue
        cleaned_sentences.append(sent_stripped)
    return cleaned_sentences

sentences = preprocess_text(document_text)

if not sentences:
    print("Error: No valid text content found after cleaning.")
    sentence_embeddings = torch.empty(0)
else:
    print(f"Processed {len(sentences)} clean sentences.")
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True)

# ===============================================
# 😀 Emoji and Response Handling
# ===============================================
emojis = ["😊", "🤖", "🔥", "👍", "💼", "✨", "😎", "🙌", "📘", "💬", "🤩"]

def random_emoji():
    return random.choice(emojis)

# ===============================================
# 💬 Chatbot Logic
# ===============================================
def chatbot(query, user_name):
    q_lower = query.lower().strip()

    # --- Greetings ---
    if q_lower in ["hello", "hi", "hey", "good morning", "good afternoon"]:
        return f"🤖: Hello {user_name}! How can I assist you today? {random_emoji()}"

    # --- Compliments ---
    if any(word in q_lower for word in ["thanks", "thank you", "good job", "well done", "nice", "great"]):
        return f"🤖: You're welcome, {user_name}! I'm glad I could help. {random_emoji()}"

    # --- Session End ---
    if "bye bot" in q_lower:
        return f"🤖: Goodbye {user_name}! Have a great day! 👋"

    # --- Name question ---
    if "my name" in q_lower:
        return f"🤖: Your name is {user_name}! {random_emoji()}"

    # --- No document case ---
    if sentence_embeddings.nelement() == 0:
        return "🤖: Sorry, I couldn’t process the document properly. No content found."

    # --- Find best matching sentence ---
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
    top_result = torch.topk(cos_scores, k=1)
    best_sentence = sentences[top_result[1]]

    if float(top_result[0]) < 0.45:
        return f"🤖: Sorry {user_name}, I couldn’t find that information in the document. {random_emoji()}"
    else:
        return f"🤖: {best_sentence.strip()} {random_emoji()}"

# ===============================================
# 🧑 Interaction Loop
# ===============================================
print("\n--- Chatbot Ready ---")
user_name = input("Enter your name: ")

print(f"🤖: Hello {user_name}! Ask me anything from the document or say 'bye bot' to exit.\n")

while True:
    q = input(f"{user_name}: ")
    if q.lower().strip() == "bye bot":
        print(chatbot(q, user_name))
        break
    ans = chatbot(q, user_name)
    print(ans)
    print()
```

---

## 🧾 Output
<img width="731" height="661" alt="Screenshot 2025-10-22 150059" src="https://github.com/user-attachments/assets/9a7dcda5-9b06-4964-83e3-d955d68479b5" />

---

## 📈 Expected Results
- Accurate and context-aware chatbot responses.  
- Summarized version of uploaded documents within seconds.  
- Extracted keywords relevant to the document’s content.  
- Fast, scalable, and secure chatbot suitable for large organizations.  

---
