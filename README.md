# RAG_Website_Scrap_QA

RAG_Website_Scrap_QA is a Python-based application that provides question-answering capabilities from web-scraped content using a Retrieval-Augmented Generation (RAG) approach.
It combines document retrieval with generative models to answer user queries accurately and contextually.

# Features

Web Scraping: Extracts website content to build a knowledge base.

Document Embedding: Converts scraped content into embeddings for efficient retrieval.

Question Answering: Generates answers based on retrieved documents using a RAG model.

Front-end Interface: app.py provides a user-facing interface for querying the system.

Database Management: Handles storage and retrieval of embeddings efficiently.

Installation
Prerequisites

Python 3.7+

Recommended: use a virtual environment.

Steps

# Clone the repository:

git clone https://github.com/Saronzeleke/RAG_Website_Scrap_QA.git
cd RAG_Website_Scrap_QA


# Install dependencies:

pip install -r requirements.txt


Set environment variables in a .env file (API keys, database URLs, etc.)

# Usage
Start the Backend (main application)
python main.py


This launches the backend API, which handles:

Web scraping

Document retrieval

Question answering

Start the Front-end Interface
python app.py


This launches the user-facing interface where users can input queries and get answers.

Web Scraping

Use the functions in scraper.py to scrape target websites.

Make sure scraping complies with the website’s terms of service.

Querying

Users input questions via the front-end.

The backend (main.py) retrieves relevant documents and generates answers using the RAG model.

# File Structure
RAG_Website_Scrap_QA/
├── main.py               # Main backend application
├── app.py                # Front-end interface
├── retrieval.py          # Document retrieval logic
├── scraper.py            # Web scraping utilities
├── db_manager.py         # Database management utilities
├── setting.py            # Configuration settings
├── requirements.txt      # Python dependencies
└── .env                  # Environment variables

# Configuration

Settings are managed in setting.py:

Database: connection parameters

Scraping: user-agent, intervals, etc.

Model: RAG parameters and API keys

# License

MIT License

# Contributing

Fork the repository, make your changes, and submit a pull request.

# Contact

GitHub - Saronzeleke
