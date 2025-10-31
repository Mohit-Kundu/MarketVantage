# MarketVantage - Technology Landscape Report Generator

**Hackathon Submission**: This project is a submission for the You.com Hackathon.

A comprehensive tool for generating technology landscape reports using AI-powered research, retrieval-augmented generation (RAG), and automated PDF creation. The application uses You.com Search API for web research and Groq/LLM APIs for content generation.

## Problem Statement

Creating comprehensive technology landscape reports is a time-consuming and resource-intensive process that typically requires:

- **Extensive Manual Research**: Analysts spend hours searching, reading, and compiling information from multiple sources
- **High Time Investment**: A single comprehensive report can take days or weeks to produce
- **Expensive Expertise**: Requires hiring experienced research analysts and technical writers
- **Difficulty Staying Current**: Technology landscapes change rapidly, making it challenging to keep reports up-to-date
- **Inconsistent Quality**: Manual compilation can lead to inconsistencies, missed information, or outdated data
- **Limited Scalability**: Producing reports for multiple technologies or topics requires significant resources

Organizations need a scalable, efficient solution to generate high-quality technology landscape reports quickly and cost-effectively.

## Solution & Outcome

MarketVantage addresses these challenges by providing an **automated, AI-powered report generation system** that:

### Key Capabilities
- **Automated Research**: Leverages You.com Search API to gather relevant information from across the web in minutes
- **Intelligent Content Retrieval**: Uses advanced RAG techniques (vector search, BM25, reranking, MMR) to find the most relevant sources
- **AI-Powered Synthesis**: Generates comprehensive, well-structured report sections using LLM APIs (Groq/Gemini)
- **Professional Output**: Automatically creates formatted PDF reports ready for presentation

### Outcomes
- **Time Savings**: Reduces report generation time from days/weeks to minutes
- **Cost Reduction**: Eliminates need for dedicated research teams for routine reports
- **Consistency**: Ensures uniform structure and quality across all generated reports
- **Scalability**: Can generate reports for any technology topic on-demand
- **Up-to-Date Information**: Leverages real-time web search to include latest developments
- **Accessibility**: Simple web interface makes advanced report generation accessible to non-technical users

### Impact
MarketVantage transforms technology landscape research from a manual, expensive process into an automated, scalable workflow, enabling organizations to make informed decisions faster and at a fraction of the traditional cost.

## Advanced RAG Techniques

MarketVantage employs a sophisticated multi-stage RAG pipeline to ensure the highest quality content retrieval:

### 1. **Multi-Query Expansion**
- Automatically generates multiple query variations from the original question
- Expands semantic coverage to capture related concepts
- Improves recall by finding relevant documents that might be missed by a single query

### 2. **Hybrid Search (Dense + Sparse)**
- **Dense Vector Search**: Uses FAISS with sentence transformers for semantic similarity matching
- **BM25 Sparse Retrieval**: Traditional keyword-based search for exact term matching
- **Best of Both Worlds**: Captures both semantic meaning and precise keyword matches

### 3. **Reciprocal Rank Fusion (RRF)**
- Intelligently combines results from dense and sparse search methods
- Weighted fusion algorithm that preserves the best candidates from both approaches
- Ensures diverse, high-quality retrieval across different matching strategies

### 4. **Cross-Encoder Reranking**
- Uses a fine-tuned cross-encoder model (`ms-marco-MiniLM-L-6-v2`) for precise relevance scoring
- Reranks top candidates with deep interaction between query and document
- Significantly improves precision by putting the most relevant content at the top

### 5. **Maximal Marginal Relevance (MMR)**
- Balances relevance and diversity in final document selection
- Prevents redundant information by ensuring diverse document coverage
- Lambda parameter (0.7) optimizes the relevance-diversity trade-off

### 6. **Windowed Retrieval**
- Expands selected chunks with neighboring context for better comprehension
- Includes surrounding text to provide full context rather than isolated fragments
- Improves LLM understanding by maintaining document flow and coherence

### Why This Matters
This multi-stage pipeline ensures that reports are built from:
- **Highly Relevant** content (reranking ensures quality)
- **Diverse** perspectives (MMR prevents redundancy)
- **Comprehensive** coverage (hybrid search captures all angles)
- **Contextually Rich** information (windowed retrieval provides full context)

The result is superior report quality compared to simple keyword or vector search alone.

## Features

- **Automated Report Generation**: Create comprehensive technology landscape reports with multiple sections
- **Advanced RAG Pipeline**: Multi-stage retrieval with hybrid search, reranking, and diversity optimization
- **Interactive UI**: Streamlit-based web interface for easy report generation
- **PDF Export**: Automatically generates professional PDF reports
- **Q&A Interface**: Interactive RAG-based question answering system with the same advanced techniques

## Prerequisites

- Python 3.8 or higher
- Windows PowerShell (for Windows users)
- API Keys:
  - **YOUCOM_API_KEY** (You.com Search API) - Required
  - **GROQ_API_KEY** (Groq LLM API) - Required for report generation
  - **GEMINI_API_KEY** (Optional) - Can be used instead of Groq

## Installation

### 1. Clone the Repository

```powershell
git clone <repository-url>
cd MarketVantage
```

### 2. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
. .\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install:
- Streamlit (web UI framework)
- Sentence Transformers (embeddings)
- FAISS (vector search)
- Groq & Google Generative AI (LLM APIs)
- ReportLab (PDF generation)
- And other dependencies

### 4. Set Up Environment Variables

Create a `.env` file in the root directory with your API keys:

```env
YOUCOM_API_KEY=your_youcom_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here  # Optional
```

**Note**: At minimum, you need `YOUCOM_API_KEY` and `GROQ_API_KEY` to generate reports.

Alternatively, you can set environment variables in PowerShell:

```powershell
$env:YOUCOM_API_KEY = "your_youcom_api_key_here"
$env:GROQ_API_KEY = "your_groq_api_key_here"
```

## Usage

### Main Application: Report Generator

Launch the Streamlit app for generating technology landscape reports:

```powershell
# Ensure virtual environment is activated
. .\.venv\Scripts\Activate.ps1

# Run the report generator app
streamlit run src/marketvantage/report_generator_app.py
```

This will:
1. Open a web browser with the Streamlit interface
2. Allow you to enter a topic (e.g., "AI Video Generation")
3. Configure report settings (embedding model, URLs per section)
4. Generate a comprehensive technology landscape report
5. Download the generated PDF report

**Report Sections Include:**
- Executive Summary
- Technology Overview
- Key Players & Market Landscape
- Recent News & Advancements
- Market Trends & Future Outlook
- Opportunities & White Space
- Conclusion

### RAG Search & Q&A Application

For interactive search and question-answering:

```powershell
# Ensure virtual environment is activated
. .\.venv\Scripts\Activate.ps1

# Run the RAG app
streamlit run src/marketvantage/rag_app.py
```

This provides:
- Vector search capabilities
- Interactive Q&A with retrieved sources
- FAISS index building and management

### Command Line Tools

#### Basic You.com Search

```powershell
python -m src.marketvantage.you_search "python typing"
python -m src.marketvantage.you_search "news this week" --count 5 --freshness week -v
```

#### News Search

```powershell
python -m src.marketvantage.you_news "technology trends"
```

#### Content Fetching

```powershell
python -m src.marketvantage.you_contents "https://example.com/article"
```


## API Documentation

- **You.com API**: https://documentation.you.com/api-reference/search
- **Groq API**: https://console.groq.com/docs
- **Gemini API**: https://ai.google.dev/docs

## Notes

- The application automatically manages FAISS indices in `data/faiss/`
- Generated PDFs are saved to `output/reports/`
- Reports are cached by topic and can be regenerated
- The application uses parallel processing for faster report generation
- Safety filters may affect content generation (more common with Gemini)

---

**Built with**: Python, Streamlit, Sentence Transformers, FAISS, Groq, You.com API
