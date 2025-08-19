# RAG_Website_Scrap_QA

RAG_Website_Scrap_QA is a Python-based application that provides question-answering capabilities from web-scraped content using a Retrieval-Augmented Generation (RAG) approach.
It combines document retrieval with generative models to answer user queries accurately and contextually.

# Features





Web Scraping: Crawls ~135 pages from visitethiopia.et (sitemap ~119 + BFS extras, ~20 min) to extract tourism information (hotels, tours, cultural sites).



Database Queries: Retrieves ~223 records from bravo_hotels, bravo_events, bravo_boats, bravo_tours, bravo_spaces, bravo_cars, and bravo_airport using FULLTEXT indexes.



RAG Pipeline: Combines web and database data to answer queries via a /ask endpoint.



Error Handling: Resolves database issues (e.g., missing description columns, invalid datetimes) for robust operation.



Production-Ready: Optimized for performance with Redis caching and asynchronous MySQL queries (aiomysql).
Prerequisites





OS: Windows (tested with PowerShell)



Python: 3.8+ (virtual environment recommended)



MySQL: 8.0.43



Redis: Latest (via Docker)



Dependencies:





httpx==0.27.2



fastapi, uvicorn, aiomysql, other dependencies in requirements.txt



Database: visitethiopia schema (from visitethiopia(13).sql)
# Installation
Clone Repository:
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
