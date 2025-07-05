

# 📄 Document Retrieval-Augmented Question Answering (RAG) System

This project is an interactive question-answering (QA) system over research papers, built with **LangChain**, **FAISS**, and **Groq LLMs**, served via a user-friendly **Streamlit** interface. It enables you to semantically search and query a collection of PDF documents with powerful retrieval-augmented generation capabilities.

---

## ⚙️ Features

 RAG (Retrieval-Augmented Generation) architecture
 PDF document ingestion
 Chunk-based text splitting
 Semantic search with FAISS vector store
 HuggingFace sentence-transformer embeddings
 Groq LLM for question answering
 Streamlit-based UI
 Real-time response with relevant document context
 Easy to extend to other document types

---

## 🛠️ Technologies Used

* **LangChain** (chains, document loaders, retrieval framework)
* **FAISS** (high-speed vector similarity search)
* **sentence-transformers** (HuggingFace embeddings on CPU for portability)
* **Groq LLM** (LLM inference using Groq's hosted infrastructure)
* **Streamlit** (interactive user interface)
* **Python** (3.12)
* **dotenv** (secure API key handling)

---

## Setup Instructions

1️⃣ **Clone the repository**

```bash
git clone https://github.com/Anudeep007-hub/Document_RAG.git
cd Document_RAG
```

2️⃣ **Set up your Python environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

3️⃣ **Install the requirements**

```bash
pip install -r requirements.txt
```

4️⃣ **Set your environment variables**

* Create a `.env` file in the project root with:

  ```bash
  GROQ_API_KEY=your_groq_api_key_here
  ```
* Replace `your_groq_api_key_here` with your actual Groq API key.

5️⃣ **Add your documents**

* Place your research papers (PDFs) in a folder named `research_paper` inside the project root.

6️⃣ **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 🧩 Usage

* Click **Generate Document Embeddings** to index your documents.
* Enter your query in the text box to perform question answering.
* Explore retrieved context using the **Document Similarity Search** expander.

---

## 💡 Architecture Overview

* PDFs are loaded and split into semantic chunks using `RecursiveCharacterTextSplitter`.
* Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2` for CPU-compatibility.
* Chunks are indexed in FAISS for efficient retrieval.
* A Groq-powered LLM (e.g., `llama3-8b-8192`) answers questions conditioned on relevant retrieved documents.
* Streamlit provides an easy, interactive front-end for end users.



## 📎 License

This project is provided under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Acknowledgements

* Groq, for fast inference LLMs
* HuggingFace, for open-source embeddings
* LangChain, for modular retrieval pipelines
* Streamlit, for effortless web interfaces


