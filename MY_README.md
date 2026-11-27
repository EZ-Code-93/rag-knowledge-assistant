# Internal Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) prototype that lets employees ask natural-language questions and receive accurate, context-grounded answers with source citations from approved internal documentation.

## Scenario & User Need

Teams often waste time searching across scattered onboarding notes, product FAQs, troubleshooting guides, and course support materials to answer routine questions. This application solves that problem by indexing a curated set of approved documents into a local vector store (ChromaDB) and pairing it with a local LLM (Ollama). Employees can quickly get clear, practical answers backed by direct references to the source files without sending internal data to external cloud services.

## Prerequisites

- **Python** `3.10.22`
- **Pipenv** (`pip install pipenv`)
- **Node.js & npm** (for the frontend client)
- **Ollama** running locally with your configured embedding and generation models pulled

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-knowledge-assistant
   ```

2. **Backend Setup**
   Navigate to the `server` directory, install dependencies with Pipenv, and configure your environment variables:
   ```bash
   cd server
   pipenv install
   ollama pull llama3.2
   ollama pull nomic-embed-text
   cp .env.example .env
   ```
   *(Review `.env` to ensure `OLLAMA_BASE_URL`, `EMBEDDING_MODEL`, `GENERATION_MODEL`, and `CLIENT_ORIGIN` match your local setup.)*

3. **Frontend Setup**
   Open a new terminal window, navigate to the `client` directory, and install the npm packages:
   ```bash
   cd client
   npm install
   ```

## Running the Application

You will need two terminal windows running simultaneously—one for the Flask backend and one for the frontend client.

### 1. Start the Backend Server
From the `server` directory:
```bash
pipenv shell
```
seed the database:
```bash
python seed_knowledge_base.py
```
run the server:
```bash
flask --app app run --debug --port 5555
```
The API will be available at `http://localhost:5555` (health check at `GET /api/health`).

### 2. Start the Frontend Client
In a separate terminal, from the project root:
```bash
cd client
npm run dev
```

## Running Tests

To run the backend unit test suite with `pytest`, ensure you are inside the `server` directory with your Pipenv environment active (or prefix with `pipenv run`):
```bash
cd server
pipenv run pytest -v
```

Basic overview of RAG workflow:
- app.py => /api/ask endpoint accepts user question and after RAG workflow return a JSON object with answer and sources
- vector_store.py retrieves embeddings
- rag_service.py builds the prompt and calls the models

Questions I used and my observations of the responses:

- I am a new employee. what steps should I take to get setup?
Response: response was grounded in company documents and relevant to the question asked. sources returned were from onboarding.txt

- A customer is having issues logging in. What steps should they take?
Answer was relevant but sources included onboarding.txt which has no connection to the question asked and if response drew from that there may be some incorrect information in the response.

- I received a strange e-mail. Should I open it?
Response: No, you should not open the email. According to the Security Guidelines, if an employee receives a suspicious email or message, they should not click links or download attachments. Instead, they should report the message through the approved security reporting process.
Answer was relevant and drew from company documentation. security guidelines were returned as source.


Future Improvements:
- workflow is quite condensed. In the future I would like to spread out the logic into a few more files to make each step a bit more modular and re-usable.

- Currently the project is closely connected to a pip environment. I personally prefer a pipenv environment and including folders for both environments clutters up the workspace a bit more then I would like. In the future it would be nice to clear out the pip folders and gear towards the use of pipenv only. 

- Some sources returned were incorrect. I would like to beef up validations and embedding scorings.