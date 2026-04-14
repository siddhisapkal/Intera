# Intera (GraphMind V2) — Technical Implementation Deep-Dive 🧠🔋

This document provides a granular architectural breakdown of the Intera platform, covering every layer from the 3D-accelerated frontend to the Graph-RAG backend logic.

---

## 1. Frontend: The Visual Intelligence Layer
Intera's UI is built as a highly responsive, zero-framework Single Page Application (SPA) designed for extreme performance and "premium" aesthetics.

### 🎨 Design Philosophy: Cyber-Glassmorphism
- **Styling System**: Driven by a centralized CSS variable system (`:root`) that defines a deep-space palette (`--surface: #0b1929`, `--accent: #3b82f6`).
- **Visual Effects**: Heavy use of `backdrop-filter: blur(16px)` and `linear-gradient` overlays to create a sense of depth and transparency.
- **Dynamic Orbs**: CSS-animated "drift" orbs provide a lively atmosphere without impacting main-thread performance.

### 🌐 3D Graph Engine (Three.js)
Intera includes a custom-coded 3D force-directed graph to visualize human memory.
- **Logic**: Implements a **D3-style force layout** in 3D space. Nodes use multi-pass physics (repulsion, attraction, and centering) to self-organize.
- **Node Geometry**: Uses `SphereGeometry` with `MeshPhongMaterial` and emissive lighting to indicate node types (e.g., User, Skill, Company).
- **Interaction**: Custom orbit controls allow the user to zoom, rotate, and pan through their "mental web."
- **Real-time Sync**: Uses an event-driven model where backend `graph_version` bumps trigger delta-updates to the 3D scene.

---

## 2. Backend: High-Performance Intelligence
The backend is a high-performance **FastAPI** application designed for complex asynchronous task handling.

### 🏗 Architecture & Isolation
- **User Namespace Strategy**: Every user is isolated at the data level. In Neo4j, nodes are linked to specific `User` nodes, and SQLite/FAISS use `user_id` partitioning.
- **Prompt Routing**: A dedicated engine decides which model (Gemini or Groq) is best suited for the current task (e.g., Groq for fast replies, Gemini for complex extraction).

### 🧠 The Signal Extraction Pipeline
This is the "brain" of the system. Every interaction goes through a multi-stage extraction:
1.  **Intent Classification**: Uses vector cosine similarity against "intent prototypes" (Prep, News, Learning) to determine search needs.
2.  **Semantic Decomposition**: An LLM extracts Triples: `(Subject)-[Relation]->(Object)`.
3.  **Confidence & Promotion**: Signals are first stored in **Ephemeral Memory** (Redis/SQLite). Only when a concept reaches a `confidence_threshold` (0.7) or is mentioned repeatedly does it get "promoted" to the **Persistent Knowledge Graph** (Neo4j).

---

## 3. Intelligence Logic
How the system makes "smart" decisions.

### 🔍 Search & RAG (Graph-RAG)
Unlike basic RAG that just searches for text, Intera performs **Semantic Retrieval**:
- **Vector Search**: FAISS retrieves related past conversation snippets.
- **Graph Traversal**: Neo4j traverses 2-degrees of freedom from the current topic to find non-obvious connections (e.g., "The user is studying Python AND is applying to Google, and Google uses Python").
- **Web Augmentation**: DuckDuckGo is scraped (via `html` and `lite` fallbacks), cleaned of noise, and summarized as "Web Facts."

### 📄 Professional Intelligence
- **Resume Parsing**: PDF streams are converted to text and passed through an LLM to extract `STRENGTH_IN`, `WORKS_AT`, and `GRADUATED_FROM` relations.
- **Gap Analysis**: The `Company Analyzer` compares the **Web Intelligence** of a company against the user's **Strength/Weakness Profile** to generate a personalized recruitment roadmap.

---

## 4. The Data Persistence Stack
| Component | Role | Why? |
| :--- | :--- | :--- |
| **Neo4j** | Knowledge Graph | Stores complex relationships that a traditional DB can't handle efficiently. |
| **FAISS** | Vector Index | Enables sub-millisecond semantic search over millions of conversation fragments. |
| **Redis** | Atomic Cache | Handles "Working Memory" and real-time state flags with extremely low latency. |
| **SQLite3** | Relational Store | Durable storage for user accounts, logs, and historical extraction metadata. |
| **Whisper (Groq)** | Audio Pipeline | Near-instant voice-to-text for mock interview mode. |

---

## 🛠 Workflow Detail: "The Lifecycle of a Message"
1.  **Intake**: Message arrives via WebSocket/REST.
2.  **Context Loading**: Graph and Vector stores are queried for "Who is this user and what do they know?"
3.  **Synthesis**: The LLM generates a response using the Loaded Context + Web Research (if enabled).
4.  **Learning**: The Extraction Engine parses the interaction and updates the **Ephemeral Store**.
5.  **Promotion**: If the user repeats something or expresses it confidently, the **Graph Service** updates Neo4j.
6.  **Visualization**: The frontend receives a `SYNC` signal and the 3D graph grows a new node.

---

> **Technical Report Summary:**
> Intera is not a chatbot; it is a **Persistent Intelligence System** that models the user's career and knowledge as a living, growing data structure.
