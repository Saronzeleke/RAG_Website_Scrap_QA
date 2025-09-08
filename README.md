# RAG_Website_Scrap_QA

## Overview

**RAG_Website_Scrap_QA** is a Python-based system that provides Retrieval-Augmented Generation (RAG) powered Question Answering (QA) over content scraped from websites. It automates the process of crawling web data, storing it efficiently, and answering user queries using advanced retrieval and language models. This project is ideal for anyone who needs to build client-specific, data-driven QA solutions on top of custom or proprietary web content.

---

## Table of Contents

- [Purpose](#purpose)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Logging](#logging)
- [Extending & Contribution](#extending--contribution)
- [License](#license)
- [Contact](#contact)

---

## Purpose

The goal of this project is to enable robust and scalable QA over web content, leveraging RAG (Retrieval-Augmented Generation) techniques. The system is designed for use cases where users or clients require accurate, contextually grounded answers sourced from specific websites.

---

## Architecture

```
+-------------------+        +----------------+        +----------------+        +------------------+
|   User Request    | -----> |    app.py      | -----> |  RAG Pipeline  | -----> |    Response      |
+-------------------+        +----------------+        +----------------+        +------------------+
                                    |                        |                             
                             +-------------+         +---------------+
                             |  crawler.py |         | retrieval.py  |
                             +-------------+         +---------------+
                                    |                        |
                             +--------------+        +-----------------+
                             | db_manager.py |<----->|    setting.py   |
                             +--------------+        +-----------------+
```

- **app.py** – Main API/server entry point; orchestrates user requests.
- **crawler.py** – Scrapes target websites and processes text.
- **db_manager.py** – Manages storage and retrieval of crawled content.
- **retrieval.py** – Implements retrieval and RAG-based QA logic.
- **setting.py** – Stores configuration and settings.
- **logging_config.yaml** – Centralized logging configuration.

---

## Core Components

### 1. Web Scraping
- `crawler.py` fetches and parses website data.
- Handles HTML extraction, text cleaning, and data chunking.

### 2. Data Storage
- `db_manager.py` abstracts the database layer.
- Supports storing raw and processed documents for efficient retrieval.

### 3. Retrieval-Augmented Generation (RAG)
- `retrieval.py` integrates retrieval models (vector search, embedding similarity) with generative LLMs for answer synthesis.
- Pipelines are modular for supporting various LLM or retriever backends.

### 4. API Layer
- `app.py` exposes endpoints for scraping, querying, and QA.
- Can be extended to RESTful or gRPC interfaces.

### 5. Configuration & Logging
- `.env` and `setting.py`: API keys and environment variables.
- `logging_config.yaml`: Flexible logging; logs stored in `qa.log`.

---

## Installation & Setup

### Prerequisites

- Python (>=3.8 recommended)
- pip

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Saronzeleke/RAG_Website_Scrap_QA.git
   cd RAG_Website_Scrap_QA
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment:**
   - Copy `.env.example` to `.env` (if provided) and fill in required fields (API keys, DB config, etc).
   - Adjust `setting.py` as needed.

---

## Configuration

- `.env`: Sensitive information like API keys.
- `setting.py`: Main application configuration (model settings, DB URLs, etc).
- `logging_config.yaml`: Logging level, format, and output destination.

---

## Usage

1. **Start the application:**
   ```bash
   python app.py
   ```

2. **Scrape a website:**
   - Use the provided API endpoint or CLI to trigger a crawl.

3. **Query the QA system:**
   - Submit questions via API (or CLI, if implemented).
   - The system retrieves relevant content and generates contextually grounded answers.

---

## Logging

- All critical operations and errors are logged as configured in `logging_config.yaml` and written to `qa.log`.
- Log levels and handlers can be customized for production or debugging.

---

## Extending & Contribution

- Modular codebase: Add new retrievers, LLM connectors, or database backends by extending `retrieval.py` and `db_manager.py`.
- Contributions should follow PEP8 and be accompanied by unit tests.
- Issues and PRs are welcome via the [GitHub repository](https://github.com/Saronzeleke/RAG_Website_Scrap_QA).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions, suggestions, or support, please contact me via email:  
📧 **sharonkuye369@gmail.com**

---

## File Structure

```
.env
.gitignore
LICENSE
README.md
__pycache__/
app.py
crawler.py
db_manager.py
logging_config.yaml
main.py
qa.log
requirements.txt
retrieval.py
setting.py
```

---

## References

- [Retrieval-Augmented Generation (RAG) Paper](https://arxiv.org/abs/2005.11401)
- [OpenAI API](https://platform.openai.com/docs/api-reference)
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)

---
