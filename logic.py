import os
import random
from typing import Any, Dict, List, Annotated
from datetime import datetime
import io
import contextlib

# Pydantic for data validation
from pydantic import BaseModel, Field

# Google GenAI for disease detection
import google.generativeai as genai

# Weather and Advisory Generation Libraries
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
from openai import AsyncOpenAI

# LangChain for the Proactive Sales Agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Initialize Google GenAI Client
try:
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        print("⚠  Warning: GOOGLE_API_KEY not found. Using mock responses for demo in /setuAgent.")
        API_KEY = "demo_mode"
    else:
        genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Warning: Google API Key not configured. The /detect endpoint will not work. Error: {e}")

# Choose the model for disease detection
DISEASE_MODEL = "gemini-1.5-flash"


# --- Disease Detection Logic ---

def perform_disease_detection(image_bytes: bytes, mime_type: str, context: str) -> str:
    """Analyzes a plant image using the Gemini model and returns the analysis."""
    try:
        model = genai.GenerativeModel(DISEASE_MODEL)
        PROMPT = f"""
        You are an agricultural expert.
        Convert the following context to English and then analyze the image:
        Context: {context}
        1. Determine if the plant is healthy or diseased.
        2. If diseased, identify the most likely plant disease.
        3. Give the final answer in Hindi.
        """
        response = model.generate_content(
            [PROMPT, {"mime_type": mime_type, "data": image_bytes}]
        )
        return response.text
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return "Could not analyze the image. Please ensure your Google API key is set correctly."



# --- Weather Data Fetching and Formatting ---

def get_weather_forecast(latitude: float, longitude: float):
    """Fetches an expanded set of 7-day weather and soil data for a specific location."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
            "relative_humidity_2m_max", "windspeed_10m_max",
            "soil_temperature_0_to_7cm_mean", "soil_moisture_0_to_7cm_mean",
            "et0_fao_evapotranspiration", "dew_point_2m_mean"
        ],
        "forecast_days": 7,
        "timezone": "auto"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_data = {
        "date": pd.to_datetime(pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ).date),
        "temperature_2m_max": daily.Variables(0).ValuesAsNumpy(),
        "temperature_2m_min": daily.Variables(1).ValuesAsNumpy(),
        "precipitation_sum": daily.Variables(2).ValuesAsNumpy(),
        "relative_humidity_2m_max": daily.Variables(3).ValuesAsNumpy(),
        "windspeed_10m_max": daily.Variables(4).ValuesAsNumpy(),
        "soil_temperature_0_to_7cm_mean": daily.Variables(5).ValuesAsNumpy(),
        "soil_moisture_0_to_7cm_mean": daily.Variables(6).ValuesAsNumpy(),
        "et0_fao_evapotranspiration": daily.Variables(7).ValuesAsNumpy(),
        "dew_point_2m_mean": daily.Variables(8).ValuesAsNumpy(),
    }
    return pd.DataFrame(data=daily_data)

def format_data_for_llm(daily_df):
    """Formats the expanded weather/soil DataFrame into a clean string for the LLM."""
    data_string = ""
    for _, day in daily_df.iterrows():
        date_str = day['date'].strftime('%A, %B %d')
        line = (
            f"- {date_str}: "
            f"Max Temp: {day['temperature_2m_max']:.1f}°C, "
            f"Rainfall: {day['precipitation_sum']:.1f} mm, "
            f"Humidity: {day['relative_humidity_2m_max']:.0f}%, "
            f"ET₀: {day['et0_fao_evapotranspiration']:.1f} mm, "
            f"Dew Point: {day['dew_point_2m_mean']:.1f}°C, "
            f"Soil Moisture: {day['soil_moisture_0_to_7cm_mean']:.3f} m³/m³\n"
        )
        data_string += line
    return data_string

# --- Dhenu 2 LLM Integration for Advisory ---

async def get_dhenu_advisory(prompt: str):
    """Connects to the Dhenu API, generates an advisory, and returns it as a string."""
    advisory_chunks = []
    async with AsyncOpenAI(
        base_url="https://api.dhenu.ai/v1",
        api_key="dh-wnQicgSa7xHf-O-LqlfdKp_dL1K3aXV68IRe7h_ITFE", # Hardcoded for testing
    ) as client:
        try:
            stream = await client.chat.completions.create(
                model="dhenu2-in-8b-preview",
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    advisory_chunks.append(chunk.choices[0].delta.content)
            return "".join(advisory_chunks)
        except Exception as e:
            error_message = f"--- An error occurred connecting to Dhenu API: {e} ---"
            print(error_message)
            return error_message

# --- Proactive Sales Agent (/setuAgent) Logic ---

# --- Comprehensive Fake Database ---
platform_rfqs: List[Dict[str, Any]] = [
    {"rfq_id": "RFQ_001", "buyer_name": "Delhi Grocers Pvt. Ltd.", "commodity": "Basmati Rice", "required_grade": "A", "quantity_tonnes": 20, "delivery_location": "Delhi", "buyer_rating": 4.8, "max_price": 4200, "urgency": "Medium", "payment_terms": "15 days", "created_date": "2025-08-15"},
    {"rfq_id": "RFQ_003", "buyer_name": "Punjab Wholesalers", "commodity": "Basmati Rice", "required_grade": "A", "quantity_tonnes": 30, "delivery_location": "Chandigarh", "buyer_rating": 4.9, "max_price": 4500, "urgency": "High", "payment_terms": "7 days", "created_date": "2025-08-16"},
    {"rfq_id": "RFQ_007", "buyer_name": "Premium Rice Exporters", "commodity": "Basmati Rice", "required_grade": "A", "quantity_tonnes": 100, "delivery_location": "Mumbai Port", "buyer_rating": 4.7, "max_price": 4800, "urgency": "Low", "payment_terms": "30 days", "created_date": "2025-08-14"},
    {"rfq_id": "RFQ_002", "buyer_name": "Mumbai Staples Co.", "commodity": "Lokwan Wheat", "required_grade": "A", "quantity_tonnes": 100, "delivery_location": "Mumbai", "buyer_rating": 4.5, "max_price": 2800, "urgency": "Medium", "payment_terms": "20 days", "created_date": "2025-08-15"},
    {"rfq_id": "RFQ_008", "buyer_name": "Haryana Flour Mills", "commodity": "Wheat", "required_grade": "A", "quantity_tonnes": 50, "delivery_location": "Panipat", "buyer_rating": 4.6, "max_price": 2600, "urgency": "High", "payment_terms": "10 days", "created_date": "2025-08-17"},
    {"rfq_id": "RFQ_004", "buyer_name": "South Indian Spice Co.", "commodity": "Turmeric", "required_grade": "A", "quantity_tonnes": 5, "delivery_location": "Chennai", "buyer_rating": 4.3, "max_price": 8500, "urgency": "Low", "payment_terms": "45 days", "created_date": "2025-08-13"},
    {"rfq_id": "RFQ_005", "buyer_name": "Gujarat Cotton Mills", "commodity": "Cotton", "required_grade": "B", "quantity_tonnes": 25, "delivery_location": "Ahmedabad", "buyer_rating": 4.4, "max_price": 5200, "urgency": "Medium", "payment_terms": "30 days", "created_date": "2025-08-16"},
    {"rfq_id": "RFQ_006", "buyer_name": "Bengal Jute Corporation", "commodity": "Jute", "required_grade": "A", "quantity_tonnes": 15, "delivery_location": "Kolkata", "buyer_rating": 4.2, "max_price": 3800, "urgency": "High", "payment_terms": "14 days", "created_date": "2025-08-17"},
    {"rfq_id": "RFQ_009", "buyer_name": "Rajasthan Pulses Ltd.", "commodity": "Chickpeas", "required_grade": "A", "quantity_tonnes": 40, "delivery_location": "Jaipur", "buyer_rating": 4.8, "max_price": 6800, "urgency": "Medium", "payment_terms": "15 days", "created_date": "2025-08-15"},
    {"rfq_id": "RFQ_010", "buyer_name": "Karnataka Coffee Traders", "commodity": "Coffee Beans", "required_grade": "A", "quantity_tonnes": 8, "delivery_location": "Bangalore", "buyer_rating": 4.9, "max_price": 12000, "urgency": "Low", "payment_terms": "21 days", "created_date": "2025-08-14"}
]

market_prices = {
    "Basmati Rice": {"current_avg": 4200, "trend": "stable", "seasonal_factor": 1.0},
    "Lokwan Wheat": {"current_avg": 2600, "trend": "rising", "seasonal_factor": 1.1},
    "Wheat": {"current_avg": 2500, "trend": "rising", "seasonal_factor": 1.1},
    "Turmeric": {"current_avg": 8200, "trend": "volatile", "seasonal_factor": 0.95},
    "Cotton": {"current_avg": 5000, "trend": "declining", "seasonal_factor": 0.9},
    "Jute": {"current_avg": 3600, "trend": "stable", "seasonal_factor": 1.05},
    "Chickpeas": {"current_avg": 6500, "trend": "rising", "seasonal_factor": 1.2},
    "Coffee Beans": {"current_avg": 11500, "trend": "volatile", "seasonal_factor": 1.1}
}


# --- Enhanced AI Tools for Sales Agent ---
@tool
def scan_for_opportunities(input_string: str) -> Dict[str, Any]:
    """Scans the marketplace for buyer requests (RFQs) matching the farmer's commodity and grade.
    Input format: 'commodity,grade' (e.g., 'Wheat,B' or 'Basmati Rice,A')
    """
    try:
        parts = input_string.split(',')
        if len(parts) != 2:
            return {"error": "Input format should be 'commodity,grade'", "matches": [], "search_attempted": True}
        
        commodity = parts[0].strip()
        grade = parts[1].strip()
    except Exception:
        return {"error": "Invalid input format", "matches": [], "search_attempted": True}
    
    print(f"\n🔍 SCANNING: Looking for '{commodity}' (Grade {grade}) opportunities...")
    
    exact_matches = [
        rfq for rfq in platform_rfqs
        if rfq.get("commodity", "").lower() == commodity.lower()
        and rfq.get("required_grade", "").lower() == grade.lower()
    ]
    
    similar_matches = [
        rfq for rfq in platform_rfqs
        if commodity.lower() in rfq.get("commodity", "").lower()
        and rfq.get("required_grade", "").lower() == grade.lower()
        and rfq not in exact_matches
    ]
    
    all_matches = exact_matches + similar_matches
    
    if all_matches:
        print(f"✅ Found {len(all_matches)} matching opportunities!")
        for rfq in all_matches:
            print(f"   - {rfq['buyer_name']}: {rfq['quantity_tonnes']}T @ {rfq['delivery_location']} (Rating: {rfq['buyer_rating']})")
    else:
        print(f"❌ No opportunities found for {commodity} Grade {grade}")
        available_commodities = list(set([rfq['commodity'] for rfq in platform_rfqs]))
        print(f"💡 Available commodities: {', '.join(available_commodities[:5])}")
    
    return {
        "matches": all_matches,
        "total_quantity": sum(rfq["quantity_tonnes"] for rfq in all_matches),
        "avg_rating": sum(rfq["buyer_rating"] for rfq in all_matches) / len(all_matches) if all_matches else 0,
        "search_attempted": True,
        "available_alternatives": list(set([rfq['commodity'] for rfq in platform_rfqs]))
    }

@tool
def analyze_market_conditions(commodity: str) -> Dict[str, Any]:
    """Analyzes current market conditions for intelligent pricing decisions."""
    print(f"\n📊 MARKET ANALYSIS: Checking conditions for {commodity}...")
    
    market_data = market_prices.get(commodity, {"current_avg": 3000, "trend": "unknown", "seasonal_factor": 1.0})
    
    analysis = {
        "commodity": commodity,
        "average_market_price": market_data["current_avg"],
        "price_trend": market_data["trend"],
        "seasonal_factor": market_data["seasonal_factor"],
        "demand_level": random.choice(["Low", "Medium", "High"]),
        "supply_level": random.choice(["Low", "Medium", "High"]),
        "recommendation": ""
    }
    
    if analysis["price_trend"] == "rising":
        analysis["recommendation"] = "Market prices are rising - good time to sell at premium"
    elif analysis["price_trend"] == "declining":
        analysis["recommendation"] = "Market prices declining - consider competitive pricing"
    else:
        analysis["recommendation"] = "Market is stable - standard pricing recommended"
    
    print(f"   📈 Average Price: ₹{analysis['average_market_price']}")
    print(f"   📊 Trend: {analysis['price_trend']}")
    print(f"   💡 Recommendation: {analysis['recommendation']}")
    
    return analysis

@tool
def calculate_smart_price(input_string: str) -> Dict[str, Any]:
    """Calculates an intelligent price using multiple factors for maximum profit."""
    try:
        parts = input_string.split(',')
        if len(parts) != 6:
            return {"error": "Input format should be 'base_price,buyer_name,rating,quantity,urgency,market_avg'"}
        
        base_price, buyer_name, buyer_rating, quantity_tonnes, urgency, market_avg = [p.strip() for p in parts]
        base_price, buyer_rating, quantity_tonnes, market_avg = float(base_price), float(buyer_rating), int(quantity_tonnes), float(market_avg)
    except Exception as e:
        return {"error": f"Invalid input format: {e}"}
    
    print(f"\n🧮 SMART PRICING: Calculating optimal price for {buyer_name}...")
    
    final_price = base_price
    adjustments = []
    
    if buyer_rating >= 4.8:
        adjustment = -30
        final_price += adjustment
        adjustments.append(f"Premium buyer discount: {adjustment}")
    elif buyer_rating < 4.0:
        adjustment = +50
        final_price += adjustment
        adjustments.append(f"Risk premium: +{adjustment}")

    if quantity_tonnes >= 50:
        adjustment = -25
        final_price += adjustment
        adjustments.append(f"Bulk order discount: {adjustment}")

    if urgency == "High":
        adjustment = +75
        final_price += adjustment
        adjustments.append(f"Urgency premium: +{adjustment}")

    final_price = round(max(final_price, base_price * 0.95), 2)
    profit_margin = ((final_price - base_price) / base_price) * 100
    
    print(f"   💰 Base Price: ₹{base_price}")
    print(f"   🎯 Final Price: ₹{final_price}")
    print(f"   📊 Profit Margin: {profit_margin:.1f}%")
    
    return {
        "final_price": final_price, "base_price": base_price, "adjustments": adjustments,
        "profit_margin": profit_margin, "market_competitive": final_price <= market_avg * 1.15
    }

@tool
def submit_quote(input_string: str) -> str:
    """Submits the final optimized quote to the selected buyer."""
    try:
        parts = input_string.split(',')
        if len(parts) != 6:
            return "❌ ERROR: Input format should be 'rfq_id,buyer_name,quote_price,farmer_name,commodity,grade'"
        
        rfq_id, buyer_name, quote_price, farmer_name, commodity, grade = [p.strip() for p in parts]
        quote_price = float(quote_price)
    except Exception as e:
        return f"❌ ERROR: Invalid input format: {e}"
    
    print(f"\n📤 SUBMITTING QUOTE: Sending offer to {buyer_name}...")
    
    rfq = next((r for r in platform_rfqs if r["rfq_id"] == rfq_id), None)
    if not rfq:
        return f"❌ ERROR: RFQ {rfq_id} not found in system."
    
    quote_id = f"QT_{random.randint(1000, 9999)}"
    estimated_response_time = random.randint(2, 48)
    
    success_message = f"""
🎉 QUOTE SUCCESSFULLY SUBMITTED!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 QUOTE DETAILS:
   • Quote ID: {quote_id}
   • Farmer: {farmer_name}
   • Commodity: {commodity} (Grade {grade})
🏢 BUYER INFORMATION:
   • Company: {buyer_name}
   • Rating: {rfq['buyer_rating']}/5.0 ⭐
   • Location: {rfq['delivery_location']}
📊 DEAL METRICS:
   • Quantity: {rfq['quantity_tonnes']} tonnes
   • Quoted Price: ₹{quote_price:,.2f}/quintal
   • Total Value: ₹{(quote_price * rfq['quantity_tonnes'] * 10):,.2f}
⏱ NEXT STEPS:
   • Expected response: {estimated_response_time} hours
   • Status: Quote submitted successfully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print("✅ Quote submitted successfully!")
    return success_message

# --- Mock LLM for Demo Mode ---
class OptimizedMockLLM:
    def invoke(self, *args, **kwargs):
        return None # This is simplified for API integration; the logic is now in optimized_demo_execution

# --- Proactive Agent Logic ---
def run_proactive_agent(farmer_details: dict):
    """Runs the intelligent proactive sales agent."""
    print(f"""
🤖 PROACTIVE AI SALES AGENT ACTIVATED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👨‍🌾 Farmer: {farmer_details['farmer_name']}
🌾 Commodity: {farmer_details['commodity']} (Grade {farmer_details['grade']})
💰 Minimum Price: ₹{farmer_details['minimum_price']:,.2f}/quintal
📍 Location: {farmer_details['location']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 Starting intelligent market analysis and sales automation...
""")
    
    tools = [scan_for_opportunities, analyze_market_conditions, calculate_smart_price, submit_quote]
    
    if API_KEY == "demo_mode":
        print("🎭 Running in DEMO MODE with optimized AI responses...\n")
        optimized_demo_execution(farmer_details, tools)
        return

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.1, google_api_key=API_KEY)
    
    prompt_template = """
You are an elite AI sales agent specializing in agricultural commodity trading. Your mission is to maximize profits for farmers by automating their entire sales process with intelligence and efficiency. IMPORTANT: If no opportunities are found in the initial scan, IMMEDIATELY terminate with a helpful Final Answer. Do NOT retry the same action multiple times.
AVAILABLE TOOLS: {tools}
RESPONSE FORMAT (Follow this EXACTLY):
Thought: [Your reasoning about the next step]
Action: [Tool name from: {tool_names}]
Action Input: [Single string with comma-separated values - NO spaces after commas]
Observation: [Tool result will appear here]
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: [Final reasoning]
Final Answer: [Comprehensive summary of achievements]
FARMER PROFILE:
• Name: {farmer_name}
• Commodity: {commodity} (Grade {grade})
• Minimum Price: ₹{minimum_price}/quintal
• Location: {location}
CRITICAL RULES:
1. If scan_for_opportunities returns empty matches, STOP immediately and provide Final Answer
2. NEVER retry the same action with the same input
3. Maximum 5 actions total - be efficient
4. Always check if previous action succeeded before proceeding
TOOL INPUT FORMATS (CRITICAL - Follow exactly):
• scan_for_opportunities: commodity,grade (e.g., "Wheat,B")
• analyze_market_conditions: commodity (e.g., "Wheat")
• calculate_smart_price: base_price,buyer_name,rating,quantity,urgency,market_avg
• submit_quote: rfq_id,buyer_name,price,farmer_name,commodity,grade
EFFICIENT WORKFLOW:
1. 🔍 SCAN ONCE: Use scan_for_opportunities with format "commodity,grade"
2. ❌ IF NO MATCHES: Immediately provide Final Answer with alternatives
3. ✅ IF MATCHES FOUND: Continue with market analysis and pricing
4. 🎯 SELECT BEST BUYER: Choose highest rated buyer
5. 📤 SUBMIT QUOTE: Complete the process
Begin the intelligent sales automation:
{input}
{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(prompt_template)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True,
        max_iterations=5, early_stopping_method="generate"
    )
    
    try:
        result = agent_executor.invoke({
            "input": "Execute the complete intelligent sales process to find and secure the best deal for my crops. If no buyers are found, provide alternatives immediately.",
            "farmer_name": farmer_details['farmer_name'], "commodity": farmer_details['commodity'],
            "grade": farmer_details['grade'], "minimum_price": farmer_details['minimum_price'],
            "location": farmer_details['location']
        })
        print(f"\n{result['output']}")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def optimized_demo_execution(farmer_details: dict, tools):
    """Optimized demo execution that handles no-data scenarios efficiently"""
    print("🎬 DEMO MODE: Running optimized agent workflow...\n")
    
    scan_input = f"{farmer_details['commodity']},{farmer_details['grade']}"
    scan_result = tools[0].func(scan_input)
    print(f"Observation: {scan_result}")
    
    if not scan_result.get('matches'):
        print(f"""
Final Answer: ❌ NO BUYERS FOUND - PROCESS OPTIMIZED
I've efficiently completed the market analysis for {farmer_details['farmer_name']}:
🔍 MARKET SCAN RESULTS:
   • Commodity: {farmer_details['commodity']} (Grade {farmer_details['grade']})
   • Buyers Found: 0 matches
💡 SMART RECOMMENDATIONS:
   • Available commodities with active buyers: {', '.join(scan_result.get('available_alternatives', [])[:3])}
   • The system will monitor 24/7 for new {farmer_details['commodity']} buyers
The process has been optimized to avoid wasted resources when no data exists.""")
        return
    
    print(f"\n🎯 PROCEEDING: Found {len(scan_result['matches'])} buyers, continuing with full process...")
    
    market_result = tools[1].func(farmer_details['commodity'])
    print(f"Observation: {market_result}")
    
    best_buyer = max(scan_result['matches'], key=lambda x: x['buyer_rating'])
    print(f"\n🎯 BEST BUYER SELECTED: {best_buyer['buyer_name']} (Rating: {best_buyer['buyer_rating']})")
    
    price_input = f"{farmer_details['minimum_price']},{best_buyer['buyer_name']},{best_buyer['buyer_rating']},{best_buyer['quantity_tonnes']},{best_buyer['urgency']},{market_result['average_market_price']}"
    price_result = tools[2].func(price_input)
    print(f"Observation: {price_result}")
    
    quote_input = f"{best_buyer['rfq_id']},{best_buyer['buyer_name']},{price_result['final_price']},{farmer_details['farmer_name']},{farmer_details['commodity']},{farmer_details['grade']}"
    quote_result = tools[3].func(quote_input)
    print(f"Observation: {quote_result}")