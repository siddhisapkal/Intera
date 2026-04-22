# 🧠 Intera: Persistent AI Career Intelligence

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Three.js](https://img.shields.io/badge/Three.js-000000?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

**Intera** is a sophisticated AI-powered career companion that doesn't just chat—it *remembers*. Powered by a sophisticated Graph-RAG (Retrieval-Augmented Generation) engine, Intera builds a persistent knowledge graph of your skills, career goals, and professional experiences.

---

## ✨ Key Features

- 🧠 **Persistent Graph Memory** — Every conversation enriches a personal knowledge graph (Neo4j), mapping your professional evolution.
- 🌐 **Real-time Web Intelligence** — Integrated DuckDuckGo search ensures your career advice is backed by current market data.
- 🏢 **Company deep-dives** — Analyze any company for interview prep, comparing their needs against your "Mental Web."
- 📄 **Smart Resume Extraction** — Upload your PDF resume to automatically populate your knowledge graph with skills and history.
- 🔮 **3D Mind Map Visualization** — Explore your professional identity through an interactive Three.js 3D force-directed graph.
- 🤖 **Hybrid LLM Engine** — Leverages the speed of **Groq (Llama 3)** and the reasoning of **Google Gemini** for optimal performance.

---

## 🏗 High-Level Architecture

Intera is built with a modular, performance-first architecture:

- **Frontend**: A zero-framework, 3D-accelerated Single Page Application (SPA) using **Three.js** and vanilla JS for a premium, low-latency experience.
- **Backend**: High-performance **FastAPI** handling asynchronous task orchestration, signal extraction, and RAG.
- **Memory Stack**:
    - **Neo4j**: Persistent Knowledge Graph for complex relationship mapping.
    - **FAISS**: Vector database for sub-millisecond semantic retrieval.
    - **SQLite3**: Durable storage for user accounts and historical logs.
- **Intelligence Pipeline**: Uses a multi-stage approach: *Intent Classification* → *Signal Extraction* → *Graph Promotion*.

---

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) & [Docker Compose](https://docs.docker.com/compose/install/)
- API Keys: [Google Gemini](https://aistudio.google.com/) and/or [Groq](https://console.groq.com/)

### Quick Start (Docker)

1. **Clone the repository**
   ```bash
   git clone https://github.com/siddhisapkal/Intera.git
   cd Intera
   ```

2. **Configure Environment**
   ```bash
   cd graphmind-v2
   cp .env.example .env
   # Add your API keys to the .env file
   ```

3. **Launch with Docker Compose**
   ```bash
   docker compose up -d
   ```

4. **Access the Application**
   Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🛠 Project Structure

```text
Intera/
├── graphmind-v2/           # Main Application Source
│   ├── backend/            # FastAPI, LLM Logic, Search, Graph Services
│   ├── frontend/           # Pure HTML/JS UI & Three.js Graph Engine
│   ├── Dockerfile          # Containerization for Backend
│   ├── docker-compose.yml  # Multi-container orchestration (App, Neo4j, Redis)
│   └── ...
├── DETAILED_ARCHITECTURAL_REPORT.md  # Deep technical dive
└── README.md               # You are here
```

---

## 💎 Design Aesthetics

Intera features a **Cyber-Glassmorphism** design:
- Deep-space palette with vibrant accent glow.
- Backdrop blurs and glass effects for a premium feel.
- Interactive 3D elements that make data feel alive.

---


*Built with ❤️ for the future of professional intelligence.*
