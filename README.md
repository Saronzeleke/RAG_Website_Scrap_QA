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

# 1 Clone the repository:

git clone https://github.com/Saronzeleke/RAG_Website_Scrap_QA.git
cd RAG_Website_Scrap_QA

# 2 Set Up Virtual Environment:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3 Install dependencies:
pip install -r requirements.txt
pip install httpx==0.27.2
# 4 Set Up Redis:
docker run -d -p 6379:6379 redis
redis-cli ping  # Expect: PONG
# 5 Set Up MySQL:
**Install MySQL 8.0.43 and log in**:
mysql -u root -p
**Create database**:
CREATE DATABASE visitethiopia;
EXIT;
**Import database**:
mysql -u root -p visitethiopia < visitethiopia(13).sql
**Database Configuration**

The visitethiopia database (109 tables) required fixes for missing description columns, slug values, and invalid datetime values (0000-00-00 00:00:00) to load ~223 records.
Schema Fixes
Issue: Missing description columns caused ERROR 1054: Unknown column 'description'. Only ~21 records loaded (from bravo_airport).
Fix: Added description columns with FULLTEXT indexes to bravo_hotels, bravo_events, bravo_boats, bravo_tours, bravo_spaces, and bravo_cars.



SQL:
USE visitethiopia;

ALTER TABLE bravo_hotels ADD description TEXT, ADD FULLTEXT(description);
ALTER TABLE bravo_events ADD description TEXT, ADD FULLTEXT(description);
ALTER TABLE bravo_boats ADD description TEXT, ADD FULLTEXT(description);
ALTER TABLE bravo_spaces ADD description TEXT, ADD FULLTEXT(description);
ALTER TABLE bravo_cars ADD description TEXT, ADD FULLTEXT(description);

-- Fix invalid datetimes in bravo_tours
SET SESSION sql_mode = 'ALLOW_INVALID_DATES';
UPDATE bravo_tours 
SET 
    start_date = NULL,
    end_date = NULL,
    publish_date = NULL,
    last_booking_date = NULL,
    created_at = NULL,
    updated_at = NULL,
    deleted_at = NULL
WHERE 
    start_date = '0000-00-00 00:00:00' OR 
    end_date = '0000-00-00 00:00:00' OR 
    publish_date = '0000-00-00 00:00:00' OR 
    last_booking_date = '0000-00-00 00:00:00' OR 
    created_at = '0000-00-00 00:00:00' OR 
    updated_at = '0000-00-00 00:00:00' OR 
    deleted_at = '0000-00-00 00:00:00';
SET SESSION sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE';
ALTER TABLE bravo_tours ADD description TEXT, ADD FULLTEXT(description);
**Data Population**

Issue: INSERT statements failed due to missing slug (ERROR 1364) and description in bravo_tours (ERROR 1054).
Fix: Added 8 rows with slug and visitethiopia.et-inspired descriptions.
SQL:
USE visitethiopia;

-- Hotels
INSERT INTO bravo_hotels (id, title, slug, content, description, status, created_at, updated_at) VALUES
(88, 'Sheraton Addis', 'sheraton-addis', 'Luxury hotel with premium amenities in Addis Ababa.', 'Located near Bole International Airport, perfect for stopovers with world-class dining and spa facilities.', 'publish', NOW(), NOW()),
(89, 'Radisson Blu Addis Ababa', 'radisson-blu-addis-ababa', 'Modern hotel in the heart of the capital.', 'Ideal for stopover travelers, offering easy access to Addis Ababa’s cultural sites and airport proximity.', 'publish', NOW(), NOW());

-- Events
INSERT INTO bravo_events (id, title, slug, content, description, status, created_at, updated_at) VALUES
(55, 'Ethiopia Tourism Vendor Expo', 'ethiopia-tourism-vendor-expo', 'Event for tourism businesses to join the industry.', 'Become a vendor: register at visitethiopia.et, submit your business license, and email info@visitethiopia.et for partnership opportunities.', 'publish', NOW(), NOW()),
(56, 'Addis Tourism Workshop', 'addis-tourism-workshop', 'Training for new tourism vendors.', 'Join Ethiopia’s tourism sector: complete the online form at visitethiopia.et and attend our vendor training sessions.', 'publish', NOW(), NOW());

-- Boats
INSERT INTO bravo_boats (id, title, slug, content, description, status, created_at, updated_at) VALUES
(21, 'Lake Tana Monastery Tour', 'lake-tana-monastery-tour', 'Boat tour visiting ancient monasteries.', 'Explore Lake Tana’s historic monasteries, perfect for a cultural stopover with scenic views.', 'publish', NOW(), NOW());

-- Tours
INSERT INTO bravo_tours (id, title, slug, content, description, status, created_at, updated_at) VALUES
(78, 'Lalibela Rock-Hewn Churches Tour', 'lalibela-rock-hewn-churches-tour', 'Visit UNESCO-listed churches carved from rock.', 'Ideal for stopovers, this tour explores Lalibela’s ancient churches, a must-see cultural gem.', 'publish', NOW(), NOW());

-- Spaces
INSERT INTO bravo_spaces (id, title, slug, content, description, status, created_at, updated_at) VALUES
(110, 'Simien Mountains National Park', 'simien-mountains-national-park', 'Hiking and wildlife in a UNESCO site.', 'A stunning stopover destination with breathtaking landscapes and rare wildlife like the Walia ibex.', 'publish', NOW(), NOW());

-- Cars
INSERT INTO bravo_cars (id, title, slug, content, description, status, created_at, updated_at) VALUES
(32, 'Bole Airport Transfer', 'bole-airport-transfer', 'Reliable shuttle service to/from Bole Airport.', 'Convenient transport for stopovers, ensuring quick and safe travel to Addis Ababa hotels.', 'publish', NOW(), NOW());

**Data Summary**:
8 new rows (2 hotels, 2 events, 1 boat, 1 tour, 1 space, 1 car)
Total records: ~223 (~20 bravo_airport + ~203 others).
Descriptions align with visitethiopia.et (e.g., Bole Airport hotels, Lalibela UNESCO sites).

# Usage
Start Server:
.venv\Scripts\Activate.ps1
uvicorn main:app --reload
# Usage
**Expected Logs**
MySQL connection pool created
Scraping: ~135 pages (~20 min, sitemap ~119 + BFS)
DB load: ~223 records (~20 bravo_airport + ~203 others)
Indexed ~358 documents (135 web + 223 DB)
Application ready
# Test Queries (PowerShell):

Hotels:
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body '{"question": "List top hotels in Addis Ababa"}'

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

**1** Fork the repository, make your changes, and submit a pull request.
**2** Add Data: Insert more records (e.g., Axum tours) with visitethiopia.et-inspired descriptions.

**3** Optimize Scraping: Enhance WebScraper class for pagination or dynamic content (see Aug 11, 2025 conversation).
**4** Improve RAG: Adjust retrival.py for better query ranking 
# Contact
GitHub - Saronzeleke
E-mail - sharonkuye369@gmail.com
