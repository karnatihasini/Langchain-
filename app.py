import os
import json
import requests
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langserve import add_routes
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# 1. API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ============================================================
# 2. TOOLS
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
        "romance": "Jab We Met, 96, Sita Ramam",
        "thriller": "Drishyam, Andhadhun, Ratsasan"
    }

    return movies.get(
        genre.lower(),
        "No movies found for that genre."
    )


@tool
def change_to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return (temp_c * 1.8) + 32


@tool
def get_weather(city: str) -> str:
    """Get current weather for an Indian city."""

    # Supported Indian cities
    indian_cities = {
        "mumbai": "Mumbai",
        "delhi": "Delhi",
        "new delhi": "New Delhi",
        "bangalore": "Bengaluru",
        "bengaluru": "Bengaluru",
        "chennai": "Chennai",
        "hyderabad": "Hyderabad",
        "kolkata": "Kolkata",
        "pune": "Pune",
        "ahmedabad": "Ahmedabad",
        "jaipur": "Jaipur",
        "surat": "Surat",
        "lucknow": "Lucknow",
        "kanpur": "Kanpur",
        "nagpur": "Nagpur",
        "indore": "Indore",
        "bhopal": "Bhopal",
        "patna": "Patna",
        "vadodara": "Vadodara",
        "coimbatore": "Coimbatore",
        "kochi": "Kochi",
        "visakhapatnam": "Visakhapatnam",
        "vijayawada": "Vijayawada",
        "mysore": "Mysuru",
        "mysuru": "Mysuru",
        "madurai": "Madurai",
        "thiruvananthapuram": "Thiruvananthapuram",
        "chandigarh": "Chandigarh",
        "agra": "Agra",
        "varanasi": "Varanasi",
        "amritsar": "Amritsar",
        "goa": "Goa"
    }

    city_clean = city.strip().lower()

    if city_clean not in indian_cities:
        return (
            "Weather information is available only "
            "for supported Indian cities."
        )

    search_city = indian_cities[city_clean]

    # --------------------------------------------------------
    # Geocoding
    # --------------------------------------------------------

    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
    )

    geo_params = {
        "name": search_city,
        "count": 1,
        "language": "en",
        "format": "json",
        "countryCode": "IN"
    }

    try:
        response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        response.raise_for_status()
        geo_data = response.json()

    except Exception as e:
        return f"Could not get location data: {e}"

    if not geo_data.get("results"):
        return f"Could not find {search_city}."

    location = geo_data["results"][0]

    latitude = location["latitude"]
    longitude = location["longitude"]

    # --------------------------------------------------------
    # Weather
    # --------------------------------------------------------

    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code",
        "temperature_unit": "celsius"
    }

    try:
        response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        response.raise_for_status()
        weather_data = response.json()

    except Exception as e:
        return f"Could not get weather data: {e}"

    current = weather_data.get("current")

    if not current:
        return "Weather data is currently unavailable."

    result = {
        "city": location.get("name", search_city),
        "country": "India",
        "temperature_celsius": current.get(
            "temperature_2m"
        ),
        "weather_code": current.get(
            "weather_code"
        )
    }

    return json.dumps(result)


# ============================================================
# 3. REGISTER TOOLS
# ============================================================

tools = [
    get_weather,
    search_movies,
    change_to_f
]


# ============================================================
# 4. GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# 5. CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a specialized AI agent restricted ONLY "
        "to Indian weather and Indian cinema.\n\n"

        "You may answer questions about:\n"
        "- Weather in Indian cities\n"
        "- Indian movies\n"
        "- Indian movie genres\n"
        "- Celsius to Fahrenheit conversion when "
        "related to weather\n\n"

        "For every other topic, respond EXACTLY with:\n"
        "I am not authorized to answer questions outside "
        "of Indian weather and cinema."
    )
)


# ============================================================
# 6. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Message for the Indian weather and cinema agent"
    )


# ============================================================
# 7. FORMAT INPUT
# ============================================================

def format_for_agent(x):

    if isinstance(x, dict):
        user_input = x["input"]
    else:
        user_input = x.input

    return {
        "messages": [
            ("user", user_input)
        ]
    }


# ============================================================
# 8. EXTRACT RESPONSE
# ============================================================

def extract_text_response(agent_output):

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:

        for value in agent_output.values():

            if (
                isinstance(value, dict)
                and "messages" in value
            ):
                messages = value["messages"]
                break

    if not messages:
        return str(agent_output)

    last_message = messages[-1]

    content = getattr(
        last_message,
        "content",
        None
    )

    if content is None:
        return str(last_message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        result = []

        for item in content:

            if isinstance(item, dict):

                if "text" in item:
                    result.append(
                        str(item["text"])
                    )

            else:
                result.append(str(item))

        return "".join(result)

    return str(content)


# ============================================================
# 9. CREATE LANGCHAIN RUNNABLE
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 10. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Indian Weather and Cinema Agent",
    description=(
        "AI Agent for Indian weather and Indian cinema"
    ),
    version="1.0.0"
)


# ============================================================
# 11. LANGSERVE ROUTES
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# ============================================================
# 12. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "status": "success",
        "message": (
            "Indian Weather and Cinema Agent "
            "is running!"
        ),
        "endpoints": {
            "docs": "/docs",
            "agent": "/agent",
            "playground": "/agent/playground/"
        }
    }


# ============================================================
# 13. HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# 14. START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )