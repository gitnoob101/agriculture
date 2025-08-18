KrishiSetu Backend 🌾
This repository contains the backend service for KrishiSetu, an AI-powered platform for farmers. Built with FastAPI, this application provides a robust API for agricultural advisories, plant disease detection, and a powerful autonomous sales agent.

✨ Features
Proactive Sales Agent: A LangChain-powered autonomous agent that scans a mock marketplace, calculates optimal prices, and submits quotes on behalf of the farmer.

AI-Powered Advisory: Integrates with the Dhenu 2 LLM to provide detailed, 7-day agricultural advisories based on hyperlocal weather and soil data.

Plant Disease Detection: Uses the Google Gemini 1.5 Flash model to analyze uploaded images of plants and provide a diagnosis in Hindi.

Robust & Scalable: Built with FastAPI and Pydantic for high performance and automatic data validation.

Cloud-Ready: Designed for easy deployment as a containerized application on services like Google Cloud Run.

🛠️ Tech Stack
Framework: FastAPI

Agent Framework: LangChain

AI Models: Google Gemini 1.5 Flash, Dhenu 2

Data Handling: Pydantic, Pandas

Deployment: Docker, Google Cloud Run

API Endpoints
The API provides three main endpoints to power the frontend application.

1. General Agricultural Advisory
Endpoint: POST /general_advisory/

Description: Fetches hyperlocal weather and soil data for a given location and generates a detailed 7-day action plan for a specific crop using the Dhenu 2 model.

Request Body:

{
  "latitude": 28.6139,
  "longitude": 77.209,
  "crop": "Wheat",
  "language": "Hindi"
}

Success Response:

{
  "advisory_report": "Weekly Focus: Critical irrigation management due to rising temperatures...\nDisease Risk Alert: Moderate risk of Powdery Mildew...\nDay-by-Day Action Plan:..."
}

2. Plant Disease Detection
Endpoint: POST /detect/

Description: Accepts an image of a plant and a text context. It uses the Gemini 1.5 Flash model to analyze the image and provide a diagnosis and recommendation in Hindi.

Request Body: multipart/form-data

image: The image file of the plant.

plant_name: (string) The name of the plant, e.g., "Tomato".

context: (string) The user's query in their native language, e.g., "ई गेहूं के फसल ह, एकर का भइल".

Success Response:

{
  "plant": "Tomato",
  "filename": "leaf_image.jpg",
  "analysis": "विश्लेषण: पौधे में पाउडरी मिल्ड्यू के लक्षण दिख रहे हैं..."
}

3. Proactive Sales Agent
Endpoint: POST /setuAgent

Description: Activates the autonomous LangChain agent. The agent scans a mock B2B marketplace, analyzes market conditions, calculates an optimal price, and submits a quote for the farmer's produce.

Request Body:

{
  "farmer_name": "Ramesh Kumar",
  "commodity": "Basmati Rice",
  "grade": "A",
  "minimum_price": 3500,
  "location": "Karnal, Haryana"
}

Success Response:

{
  "agent_run_log": "🤖 PROACTIVE AI SALES AGENT ACTIVATED...\n\n🔍 SCANNING: Looking for 'Basmati Rice' (Grade A) opportunities...\n✅ Found 3 matching opportunities!\n...etc."
}

🚀 Setup and Local Development
1. Prerequisites
Python 3.10 or higher

pip and venv

2. Installation
Clone the repository:

git clone git@github.com:gitnoob101/agriculture.git
cd krishiSetuBackend

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Create a requirements.txt file with the following content:

fastapi
uvicorn[standard]
pydantic
python-dotenv
google-generativeai
openmeteo-requests
requests-cache
retry-requests
pandas
openai
langchain
langchain-core
langchain-google-genai

Install the dependencies:

pip install -r requirements.txt

3. Environment Variables
Create a file named .env in the root of the project.

Add your Google API key to this file:

GOOGLE_API_KEY="your_google_api_key_here"

4. Running the Server
Start the development server using Uvicorn:

uvicorn main:app --reload

The API will be available at http://127.0.0.1:8000.

☁️ Deployment to Google Cloud Run
This application is designed to be deployed as a container on Google Cloud Run.

Create a Dockerfile in your project root (refer to the one provided in our conversation).

Build the Docker image and push it to Google Artifact Registry:

gcloud builds submit --tag gcr.io/your-gcp-project-id/krishisetu-backend

Deploy the image to Cloud Run:

gcloud run deploy krishisetu-backend \
  --image gcr.io/your-gcp-project-id/krishisetu-backend \
  --platform managed \
  --region asia-south2 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY="your_google_api_key_here"
