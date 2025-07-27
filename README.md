✅ Overview
Combines traditional scraping (BeautifulSoup) with LLMs (GPT-4)

Extracts structured data (e.g., job titles, prices, articles) from messy HTML

Outputs clean, structured JSON

📦 Features
Web scraping using requests + BeautifulSoup

LLM-powered extraction with OpenAI GPT-4

Handles nested/inconsistent HTML

JSON-formatted output for easy integration

Prompt-based semantic understanding

🛠️ Tech Stack
requests – Fetch web content

beautifulsoup4 – Parse HTML

openai – GPT-4 API

dotenv – Store API key

json – Parse LLM output

📁 Files
llm_scraper.ipynb – Main notebook

.env – API key storage

requirements.txt – Dependencies

README.md – Project info

⚙️ How It Works
Scrape page → clean text

Prompt GPT-4 to extract structured info

Parse and display/store JSON

🔍 Prompting Strategy
Clear role: "You are a data extractor"

Specific instructions: "Extract job title, location..."

JSON output format specified

🧠 Handles Edge Cases
✅ Nested HTML → flattened to plain text

✅ JS-loaded content → extendable with Selenium

✅ Label inconsistencies → LLM uses context

🚧 Limitations
API cost and rate limits

Possible hallucinated or malformed JSON

Not ideal for large/token-heavy pages
