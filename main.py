import os
import httpx
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

try:
    # Configure the Gemini API key
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # Get the data.gov.in API key
    DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")
    
    # Check if keys were actually loaded
    if not os.getenv("GOOGLE_API_KEY") or not DATA_GOV_API_KEY:
        raise KeyError
        
except KeyError:
    raise EnvironmentError(
        "API keys not found. Please create a .env file and add GOOGLE_API_KEY and DATA_GOV_API_KEY."
    )

class QuestionRequest(BaseModel):
    question: str

app = FastAPI(
    title="Agricultural Data Backend",
    description="An API that answers agricultural questions by fetching live market data.",
    version="2.0.0",
)

extraction_schema = {
    "type": "OBJECT",
    "properties": {
        "district": {"type": "STRING", "description": "The district mentioned in the user's query."},
        "commodity": {"type": "STRING", "description": """Your task is to extract the district and the commodity from a user's query.
The user is in India and might ask in native Indian languages (e.g., Hindi) or Hinglish.
The commodity name might not be in English. If it's not, you MUST translate it to its standard English name.
For example, if the query is 'Ballia mein aloo ka bhav kya hai?', you should extract 'Ballia' as the district and 'Potato' as the commodity.
"""}
    },
    "required": ["district", "commodity"]
}

generation_config = {
    "response_mime_type": "application/json",
    "response_schema": extraction_schema,
}

model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config=generation_config
)

@app.post("/ask/")
async def process_question(request: QuestionRequest):
    user_question = request.question
    print(f"Received question: {user_question}")

    try:
        prompt = f"Extract the district and commodity from the following user query: '{user_question}'"
        response = await model.generate_content_async(prompt)
        
        # Correctly parse the JSON string from the response's text part
        response_text = response.candidates[0].content.parts[0].text
        entities = json.loads(response_text)

        district = entities.get("district", "").lower()
        commodity = entities.get("commodity", "").lower()

        if not district or not commodity:
            raise HTTPException(status_code=400, detail="Could not identify a district and commodity from your question. Please be more specific.")

        print(f"Extracted entities: District='{district}', Commodity='{commodity}'")

        api_url = (
            f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={DATA_GOV_API_KEY}&format=json&filters%5Bdistrict%5D={district}&filters%5Bcommodity%5D={commodity}"
        )

        async with httpx.AsyncClient() as client:
            gov_response = await client.get(api_url)
            gov_response.raise_for_status()
            data = gov_response.json()

        records = data.get("records", [])
        if not records:
            answer = f"I couldn't find any data for '{commodity}' in the '{district}' district."
        else:
            record = records[0]
            answer = (
                f"Here is the latest price for {record['commodity']} ({record['variety']}) "
                f"in {record['market']}, {record['district']}:\n"
                f"- Minimum Price: ₹{record['min_price']}\n"
                f"- Maximum Price: ₹{record['max_price']}\n"
                f"- Modal Price: ₹{record['modal_price']}\n"
                f"(Data from {record['arrival_date']})"
            )

        return {"original_question": user_question, "answer": answer}

    except httpx.HTTPStatusError as e:
        print(f"Error calling government API: {e}")
        raise HTTPException(status_code=502, detail="Failed to retrieve data from the external source.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Backend is running!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
