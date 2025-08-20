# KrishiSetu Backend 🌾

This repository contains the backend service for **KrishiSetu**, an AI-powered platform designed to empower farmers. Built with FastAPI, this application provides a robust API for agricultural advisories, plant disease detection, and a powerful autonomous sales agent.

-----

## ✨ Features

  - **Proactive Sales Agent**: A LangChain-powered autonomous agent that scans a mock marketplace, calculates optimal prices, and submits quotes on behalf of the farmer.
  - **AI-Powered Advisory**: Integrates with the Dhenu 2 LLM to provide detailed, 7-day agricultural advisories based on hyperlocal weather and soil data.
  - **Plant Disease Detection**: Uses the Google Gemini 1.5 Flash model to analyze uploaded images of plants and provide a diagnosis in Hindi.
  - **Robust & Scalable**: Built with FastAPI and Pydantic for high performance and automatic data validation.
  - **Cloud-Ready**: Designed for easy deployment as a containerized application on services like Google Cloud Run.

-----

## 🛠️ Tech Stack

  - **Framework**: FastAPI
  - **Agent Framework**: LangChain
  - **AI Models**: Google Gemini 1.5 Flash, Dhenu 2
  - **Data Handling**: Pydantic, Pandas
  - **Deployment**: Docker, Google Cloud Run

-----

## 📂 Project Structure

The repository is organized as follows:

```
.
├── .dockerignore      # Specifies files to ignore in the Docker build
├── .env               # Stores environment variables (e.g., API keys)
├── .gitignore         # Specifies files for Git to ignore
├── Dockerfile         # Instructions for building the Docker container
├── logic.py           # Core application logic (AI models, agent tools)
├── main.py            # FastAPI application entrypoint and API endpoints
├── README.md          # This documentation file
└── requirements.txt   # Python project dependencies
```

-----

## 🔌 API Endpoints

The API provides three main endpoints to power the frontend application.

### 1\. General Agricultural Advisory

  - **Endpoint**: `POST /general_advisory/`
  - **Description**: Fetches hyperlocal weather and soil data for a given location and generates a detailed 7-day action plan for a specific crop using the Dhenu 2 model.
  - **Request Body**:
    ```json
    {
      "latitude": 28.6139,
      "longitude": 77.209,
      "crop": "Wheat",
      "language": "Hindi"
    }
    ```

### 2\. Plant Disease Detection

  - **Endpoint**: `POST /detect/`
  - **Description**: Accepts an image of a plant and a text context. It uses the Gemini 1.5 Flash model to analyze the image and provide a diagnosis and recommendation in Hindi.
  - **Request Body**: `multipart/form-data`
      - `image`: The image file of the plant.
      - `plant_name`: (string) The name of the plant, e.g., "Tomato".
      - `context`: (string) The user's query in their native language.

### 3\. Proactive Sales Agent

  - **Endpoint**: `POST /setuAgent`
  - **Description**: Activates the autonomous LangChain agent. The agent scans a mock B2B marketplace, analyzes market conditions, calculates an optimal price, and submits a quote for the farmer's produce.
  - **Request Body**:
    ```json
    {
      "farmer_name": "Ramesh Kumar",
      "commodity": "Basmati Rice",
      "grade": "A",
      "minimum_price": 3500,
      "location": "Karnal, Haryana"
    }
    ```

-----

## 🚀 Setup and Local Development

### 1\. Prerequisites

  - Python 3.10 or higher
  - `pip` and `venv`

### 2\. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/gitnoob101/agriculture.git
    cd agriculture
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```
3.  **Install dependencies from `requirements.txt`:**
    ```bash
    pip install -r requirements.txt
    ```

### 3\. Environment Variables

1.  Create a file named `.env` in the root of the project.
2.  Add your Google API key to this file:
    ```env
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

### 4\. Running the Server

Start the development server using Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

Once the server is running, you can access the **interactive API documentation** by navigating to `http://127.0.0.1:8000/docs` in your browser. From there, you can test all the endpoints directly.

-----

## ☁️ Deployment to Google Cloud Run

This application is designed to be deployed as a container on Google Cloud Run.

1.  **Build the Docker image** and push it to Google Artifact Registry:

    ```bash
    gcloud builds submit --tag gcr.io/your-gcp-project-id/krishisetu-backend
    ```

2.  **Deploy the image** to Cloud Run:

    ```bash
    gcloud run deploy krishisetu-backend \
      --image gcr.io/your-gcp-project-id/krishisetu-backend \
      --platform managed \
      --region asia-south2 \
      --allow-unauthenticated \
      --set-env-vars GOOGLE_API_KEY="your_google_api_key_here"
    ```