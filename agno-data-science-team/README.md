# Agno Data Science Team 🤖📊

A multi-agent AI system built using the **Agno** framework, designed to assist with end-to-end Machine Learning and Data Science pipelines. This project features specialized AI agents collaborating under an Agent OS to load data, perform exploratory data analysis (EDA), generate visualizations, write Python code, and execute shell commands.

## 🌟 Key Features

*   **Multi-Agent Architecture**: Uses a specialized team of agents for different stages of the data science workflow.
*   **Data Ingestion & Management**: Agents dedicated to loading CSVs and managing the filesystem.
*   **Automated EDA**: Pandas-powered agent for summarizing data, checking shapes, and info.
*   **Intelligent Visualization**: Generates matplotlib charts (bar, pie, line, histogram, scatter) based on natural language requests.
*   **Code Generation & Execution**: A coding agent that can write ML code (sklearn, pandas, numpy) and a shell agent to run it safely.
*   **Web Search Capability**: Integrated DuckDuckGo search for documentation and external knowledge retrieval.
*   **Agent OS Interface**: Hosted as a FastAPI application with session state memory and SQLite storage.

## 👥 The Agent Team

*   **Data Loader Agent**: Reads and samples CSV files efficiently without overloading context.
*   **File Manager Agent**: Navigates and manages project files and structures.
*   **Data Understanding Agent**: Explores data using Pandas (`head`, `tail`, `describe`, `value_counts`).
*   **Visualization Agent**: Creates relevant visual plots based on the data provided and saves them.
*   **Coding Agent**: Writes specific Python code for ML tasks like data cleaning, feature engineering, and training.
*   **Shell Agent**: Executes Python files and environment management commands (e.g., `uv add`).

## 🚀 Getting Started

### Prerequisites

*   Python 3.10+
*   [uv](https://github.com/astral-sh/uv) (for faster Python package management)
*   OpenAI API Key (or other supported LLM provider)

### Installation

1.  **Clone the repository and navigate to the project:**
    ```bash
    git clone <your-repo-url>
    cd agno-data-science-team
    ```

2.  **Install dependencies using `uv`:**
    ```bash
    uv sync
    ```

3.  **Environment Setup:**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    ```

### Running the App

Start the Agent OS FastAPI server:
```bash
python app.py
```
Alternatively, test the default script with:
```bash
python main.py
```

The application will launch and be ready to accept data science tasks!

## 📁 Project Structure

*   `app.py`: Main Agent OS and multi-agent setup, serving as a FastAPI app.
*   `main.py`: Simple entry point script.
*   `data/`: Directory for placing your datasets (e.g., `car_details.csv`).
*   `plots/`: Directory where the Visualization Agent saves generated charts.
*   `src/`: Directory for any generated source files.
*   `models/`: Directory for saving trained ML models.
*   `reports/`: Directory for generated analysis reports.
