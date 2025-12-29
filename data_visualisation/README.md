# Data Visualisation

A multi-agent analytics system that uses natural language queries to extract sales data and generate charts.

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Virtual Environment

**Windows (Command Prompt):**
```cmd
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python main.py
```

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) running locally with the `qwen3:32b` model

Make sure Ollama is running at `http://localhost:11434` before starting the application.
