# --- PART 0: All Library Imports ---

# Standard Libraries

from typing import Annotated
import io
import contextlib

# FastAPI and Uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# Pydantic for data validation
from pydantic import BaseModel, Field

from dotenv import load_dotenv

import logic

# --- PART 1: FastAPI App Initialization ---

load_dotenv()
app = FastAPI(
    title="KrishiSetu",
    description="An API for farmers providing query handling, disease detection, weather information, and a proactive sales agent.",
    version="1.1.0",
)

origins = ["http://localhost:5173","https://krishisetu-a6f6f.web.app"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



# --- Pydantic Models for API Input Validation ---

class Location(BaseModel):
    """A model to represent geographical coordinates, crop, and language."""
    latitude: float = Field(..., examples=[28.6139], description="Latitude of the location.")
    longitude: float = Field(..., examples=[77.2090], description="Longitude of the location.")
    crop: str = Field("Rice", examples=["Wheat"], description="The crop for which the advisory is needed.")
    language: str = Field("English", examples=["Hindi"], description="The language for the advisory output.")

class QueryInput(BaseModel):
    """Input model for the /query endpoint."""
    question: str = Field(..., examples=["What is the best fertilizer for wheat?"], description="The user's question.")
    location: Location

# New model for the /setuAgent endpoint
class FarmerProfile(BaseModel):
    """Input model for the proactive sales agent."""
    farmer_name: str = Field(..., examples=["Ramesh Kumar"], description="The farmer's full name.")
    commodity: str = Field(..., examples=["Basmati Rice"], description="The commodity the farmer is selling.")
    grade: str = Field(..., examples=["A"], description="The grade of the commodity (e.g., A, B, C).")
    minimum_price: float = Field(..., examples=[3500.0], description="The farmer's minimum acceptable price per quintal.")
    location: str = Field(..., examples=["Karnal, Haryana"], description="The farmer's location.")


# --- API Endpoints ---

@app.get("/")
def read_root():
    """A default root endpoint to welcome users."""
    return {"message": "Welcome to the KrishiSetu Farmer AI Assistant API!"}

@app.post("/detect/")
async def detect_plant_disease(
    image: Annotated[UploadFile, File(description="An image of the plant.")],
    plant_name: Annotated[str, Form(description="The name of the plant.", examples=["Tomato"])],
    context: Annotated[str, Form(description="Context about the crop in Hindi", examples=["ई गेहूं के फसल ह, एकर का भइल"])]
):
    """Accepts an image and context for disease detection."""
    print(f"Received image '{image.filename}' for plant: '{plant_name}'.")
    image_bytes = await image.read()
    analysis_text = logic.perform_disease_detection(
        image_bytes=image_bytes,
        mime_type=image.content_type,
        context=context
    )
    return {"plant": plant_name, "filename": image.filename, "analysis": analysis_text}


@app.post("/general_advisory/")
async def get_general_advisory(location: Location):
    """Provides a detailed agricultural advisory for a given location and crop."""
    print(f"🛰  Fetching enhanced forecast for {location.crop} at Lat {location.latitude}, Lon {location.longitude}...")
    try:
        farm_data = logic.get_weather_forecast(location.latitude, location.longitude)
    except Exception as e:
        error_message = f"Fatal Error: Could not fetch weather data. Error: {e}"
        print(error_message)
        return {"error": error_message}

    print("📝  Formatting data for the AI...")
    formatted_data = logic.format_data_for_llm(farm_data)

    master_prompt = f"""
    You are Dhenu, a Chief Agronomist AI.
    FARM CONTEXT:
    - Location: Latitude {location.latitude}, Longitude {location.longitude}
    - Crop: {location.crop}
    - Time of Year: Peak Kharif Season
    - Output Language: {location.language}
    7-DAY FORECAST & SOIL DATA:
    {formatted_data}
    YOUR TASK:
    Analyze all data to create a practical advisory. Follow the output structure below exactly.
    OUTPUT STRUCTURE:
    Weekly Focus: [Your single, bold sentence on the week's most critical issue]
    Disease Risk Alert: [Your identification of the most likely disease, or "No major disease risk this week"]
    Day-by-Day Action Plan:
    [Day Name], [Date]
    - Irrigation: [Your advice]
    - Fertilizer Application: [Your advice]
    - Pest/Disease Management: [Your advice]
    ... (continue for all 7 days)
    Generate the advisory now in {location.language}.
    """

    print(f"🧠  Sending data to Dhenu 2 for expert analysis...")
    advisory = await logic.get_dhenu_advisory(master_prompt)
    print("✅  Advisory complete.")

    return {"advisory_report": advisory}


@app.post("/setuAgent")
async def run_sales_agent(profile: FarmerProfile):
    """
    Activates the Proactive AI Sales Agent to find market opportunities,
    calculate optimal prices, and submit quotes automatically.
    """
    # Convert the Pydantic model to a dictionary for the agent function
    farmer_details = profile.model_dump()
    
    # Use StringIO and redirect_stdout to capture all print statements from the agent's run
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        # The agent function is synchronous. In a high-traffic production app,
        # you might run this in a thread pool to avoid blocking.
        # For this integration, a direct call is sufficient.
        logic.run_proactive_agent(farmer_details)
        
    # Get the captured output as a string
    agent_output = output_buffer.getvalue()
    
    return {"agent_run_log": agent_output}