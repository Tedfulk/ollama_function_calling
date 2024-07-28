import json
import ollama
import asyncio
from typing import Dict


def get_flight_times(departure: str, arrival: str) -> str:
    flights = {
        "NYC-LAX": {
            "departure": "08:00 AM",
            "arrival": "11:30 AM",
            "duration": "5h 30m",
        },
        "LAX-NYC": {
            "departure": "02:00 PM",
            "arrival": "10:30 PM",
            "duration": "5h 30m",
        },
        "LHR-JFK": {
            "departure": "10:00 AM",
            "arrival": "01:00 PM",
            "duration": "8h 00m",
        },
        "JFK-LHR": {
            "departure": "09:00 PM",
            "arrival": "09:00 AM",
            "duration": "7h 00m",
        },
        "CDG-DXB": {
            "departure": "11:00 AM",
            "arrival": "08:00 PM",
            "duration": "6h 00m",
        },
        "DXB-CDG": {
            "departure": "03:00 AM",
            "arrival": "07:30 AM",
            "duration": "7h 30m",
        },
    }

    key = f"{departure}-{arrival}".upper()
    return json.dumps(flights.get(key, {"error": "Flight not found"}))


def get_weather_forecast(city: str) -> str:
    weather_data: Dict[str, Dict[str, str]] = {
        "NEW YORK": {"temperature": "72°F", "condition": "Partly cloudy"},
        "LOS ANGELES": {"temperature": "75°F", "condition": "Sunny"},
        "LONDON": {"temperature": "62°F", "condition": "Rainy"},
        "PARIS": {"temperature": "68°F", "condition": "Clear"},
        "SYDNEY": {"temperature": "70°F", "condition": "Windy"},
    }

    forecast = weather_data.get(city.upper(), {"error": "City not found"})
    return json.dumps(forecast)


async def run(model: str):
    client = ollama.AsyncClient()
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Paris today?",
        }
    ]
    # messages = [{'role': 'user', 'content': 'What is the flight time from New York (NYC) to Los Angeles (LAX)?'}]

    response = await client.chat(
        model=model,
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_flight_times",
                    "description": "Get the flight times between two cities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "departure": {
                                "type": "string",
                                "description": "The departure city (airport code)",
                            },
                            "arrival": {
                                "type": "string",
                                "description": "The arrival city (airport code)",
                            },
                        },
                        "required": ["departure", "arrival"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": "Get the weather forecast for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The name of the city",
                            },
                        },
                        "required": ["city"],
                    },
                },
            },
        ],
    )

    messages.append(response["message"])

    if not response["message"].get("tool_calls"):
        print("The model didn't use any function. Its response was:")
        print(response["message"]["content"])
        return

    if response["message"].get("tool_calls"):
        available_functions = {
            "get_flight_times": get_flight_times,
            "get_weather_forecast": get_weather_forecast,
        }
        for tool in response["message"]["tool_calls"]:
            function_name = tool["function"]["name"]
            function_to_call = available_functions[function_name]

            arguments = tool["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if function_name == "get_flight_times":
                function_response = function_to_call(
                    arguments["departure"],
                    arguments["arrival"],
                )
            elif function_name == "get_weather_forecast":
                function_response = function_to_call(arguments["city"])

            messages.append(
                {
                    "role": "tool",
                    "content": function_response,
                }
            )

    final_response = await client.chat(model=model, messages=messages)
    print(final_response["message"]["content"])


asyncio.run(run("mistral"))
