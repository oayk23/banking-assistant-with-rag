# 💳 Banking Assistant – RAG-Powered AI for Smarter Customer Support

**Banking Assistant** is an intelligent, retrieval-augmented generation (RAG) system designed to streamline and enhance customer support experiences in the banking domain. Built with a blend of modern NLP techniques and LLM integration, this assistant delivers accurate, context-aware responses to user queries by retrieving the most relevant documentation snippets from a vectorized knowledge base.

---

## 🚀 Project Highlights

- 🔍 **RAG (Retrieval-Augmented Generation):** Combines semantic search with generative AI to produce informed, real-time answers.
- 🧠 **FAISS Vector Store:** Fast similarity search over embeddings to fetch contextually relevant documents.
- 🤖 **Transformers:** Utilizes HuggingFace models for both embedding generation and response generation.
- 📂 **Intent Classification:** Routes user inputs to the right sub-knowledge base based on their detected intent.
- 🔧 **Modular Pipeline:** Each step – from preprocessing and indexing to querying and response generation – is modularized for easy experimentation and extension.
- 🌐 **API-Ready:** Interact with the assistant through a simple RESTful interface.

---

## ⚙️ Setup Instructions

You can set up the project easily by running the platform-specific setup script.  

### On Linux / macOS:
```bash
bash setup.sh
```
### On Windows:
```bash
setup.bat
```
These scripts will install necessary dependencies and set up the environment for smooth execution.
---
## 🧪 How It Works
1. Instruction Normalization: User prompts are cleaned and normalized for clarity.

2. Intent Classification: A lightweight classifier determines the intent category (e.g., card issues, account access, loan queries).

3. Vector Search with FAISS: Based on the intent, a corresponding FAISS index is queried for relevant documents.

4. Contextual Generation: The top-k relevant documents are passed as context to a transformer-based LLM, which generates the final response.

## ⚠️ Limitations
- 💾 Memory Usage: Embedding and querying large-scale document corpora may cause memory spikes. Batch size and model selection should be adjusted accordingly for limited environments.

- 🧠 Model Size Constraints: Low-parameter models may struggle with nuanced generation or instruction-following. Higher-performance results are achievable with larger models (e.g., Mistral, LLaMA, or GPT variants).

- 🗃️ Static Corpus: The current system operates over a static indexed corpus. Real-time document updates require re-indexing.

- 🌐 No External API Integration Yet: This version doesn't fetch real-time banking data (e.g., account balances, transactions).

## 🔌 API Usage
The system is accessible via an API endpoint. Here’s a basic example of how to interact:
```
POST /chat
Content-Type: application/json

{
  "query": "My card got stuck in the ATM, what should I do?"
}
Response:
```
```
{
  "response": "I understand how frustrating it can be to have your card swallowed by an ATM. Here's what you can do..."
}
```
The API layer allows easy integration with chat interfaces, mobile apps, or internal support tools.

## 🌟 Future Directions
- Real-time document ingestion and indexing

- User feedback loop for response refinement

- Fine-tuned intent classifier

- Multi-language support

## 📁 Folder Structure Overview
```
banking-assistant/
├── setup.sh / setup.bat      # Setup scripts for dependencies
├── banking_assistant/
│   ├── embedder.py           # Embedding logic using transformers
│   ├── correcter.py          # grammar and spelling correcter module
│   ├── classifier/           # Intent classification module
|       ├── train.py
|       ├── inference.py      
│   ├── llm.py                # LLM pipeline
│   ├── full_pipeline.py      # Full Pipeline
├── data_preparation/         # Data Preparation Package
│   ├── correct_instructions.py # Instruct Correcter
├── index_preparation/        # Index Preparation Package
│   ├── index_preparation.py  # Prepares Vector Embeddings
├── app.py                    # Restful api with FastAPI
├── main.py                   # main.py file that prepares indexes,classifiers etc.
├── inference.py              # inference file
```
## ✨ Final Notes
Banking Assistant demonstrates how retrieval-augmented LLM systems can be harnessed for domain-specific applications with precision and efficiency. Whether integrated into an enterprise solution or used as a research base, it showcases a clean architecture ready for real-world challenges.

Feel free to fork, extend, and contribute!

## 📬 Contact
For issues, suggestions, or collaboration opportunities, feel free to open an issue or reach out.
