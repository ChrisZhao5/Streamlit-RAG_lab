# 📄 Streamlit Hybrid RAG Lab: Privacy-First Document Q&A

## 📌 Project Overview
This repository contains `app_stable.py`, a working prototype of a **Hybrid Retrieval-Augmented Generation (RAG) System** built with **Streamlit** and LangChain. 

I architected this project to address a critical business challenge in highly regulated industries like Insurance: **How can we leverage the reasoning power of Large Language Models (LLMs) to parse massive documents (e.g., policy contracts, medical histories) without compromising data privacy or risking AI hallucinations?**

## 🎯 Business Value & Use Case
* **Underwriting Efficiency:** Designed to help underwriters instantly retrieve specific clauses from dense policy documents, drastically reducing manual review time.
* **Driving Straight-Through Processing (STP):** By automating document Q&A with a high degree of confidence and factual accuracy, this tool acts as a stepping stone toward higher STP rates in policy approvals.
* **100% Data Privacy:** Sensitive documents are processed and embedded **locally**. Proprietary data never leaves the local environment to train public models.

## 🛠️ Core Architecture
Instead of a basic API wrapper, this system uses a hybrid pipeline to balance cost, privacy, and performance:

* **Frontend:** Built with **Streamlit** for a seamless, interactive user interface that allows non-technical users to upload PDFs and query them instantly.
* **Local Embeddings:** Uses HuggingFace's `Sentence-Transformers` to run vectorization strictly on the local CPU, ensuring maximum data security.
* **Vector Database:** Implements **FAISS** (Facebook AI Similarity Search) for lightning-fast semantic retrieval of relevant document chunks.
* **Zero-Hallucination Prompting:** The inference LLM is configured with `Temperature = 0.0` and strict instructions to *only* answer based on the retrieved context, forcing it to cite sources or admit if the answer isn't in the text.

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/ChrisZhao5/Streamlit-RAG_lab.git](https://github.com/ChrisZhao5/Streamlit-RAG_lab.git)
cd Streamlit-RAG_lab
