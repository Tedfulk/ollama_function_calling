# Ollama Function Calling Demo

This Demo demonstrates how to use a function calling model to fetch flight times between different cities. The table below lists the models that support function calling along with their parameter options:

| Model Name               | Supports Function Calling | Parameter Options      |
|--------------------------|---------------------------|------------------------|
| llama3.1                 | Yes                       | 8B, 70B, 405B          |
| mistral-nemo             | Yes                       | 12B                    |
| firefunction-v2          | Yes                       | 70B                    |
| llama3-groq-tool-use     | Yes                       | 8B, 70B                |
| command-r-plus           | Yes                       | 104B                   |
| mixtral                  | Yes                       | 8x7B, 8x22B            |
| mistral                  | Yes                       | 7B                     |

## Overview

This code uses an asynchronous client from the `ollama` library to interact with a language model that supports function calling. The model can call a predefined function, `get_flight_times`, to fetch flight times between two cities based on their airport codes.

### Features

- **Function Calling:** The model can call the `get_flight_times` function to fetch flight information.
- **Asynchronous Execution:** Utilizes `asyncio` for asynchronous API calls.
- **Conversation Handling:** Manages a conversation history to maintain context between API calls.

## Code Explanation

### Dependencies

Ensure you have the `ollama` library installed. You can clone the repo and install it using poetry:

```bash
poetry install
```

or install it using pip and copy the code from `main.py` to your project:

```bash
pip install ollama
```

### Flight Times Function

The `get_flight_times` function simulates an API call to get flight times between two cities. In a real application, this would fetch data from a live database or API.

```python
def get_flight_times(departure: str, arrival: str) -> str:
    flights = {
        'NYC-LAX': {'departure': '08:00 AM', 'arrival': '11:30 AM', 'duration': '5h 30m'},
        'LAX-NYC': {'departure': '02:00 PM', 'arrival': '10:30 PM', 'duration': '5h 30m'},
        'LHR-JFK': {'departure': '10:00 AM', 'arrival': '01:00 PM', 'duration': '8h 00m'},
        'JFK-LHR': {'departure': '09:00 PM', 'arrival': '09:00 AM', 'duration': '7h 00m'},
        'CDG-DXB': {'departure': '11:00 AM', 'arrival': '08:00 PM', 'duration': '6h 00m'},
        'DXB-CDG': {'departure': '03:00 AM', 'arrival': '07:30 AM', 'duration': '7h 30m'},
    }

    key = f'{departure}-{arrival}'.upper()
    return json.dumps(flights.get(key, {'error': 'Flight not found'}))
```

### Main Asynchronous Function

The `run` function initializes a conversation with a user query, sends the query and function description to the model, processes any function calls made by the model, and gets the final response.

```python
async def run(model: str):
    client = ollama.AsyncClient()
    messages = [{'role': 'user', 'content': 'What is the flight time from New York (NYC) to Los Angeles (LAX)?'}]

    response = await client.chat(
        model=model,
        messages=messages,
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'get_flight_times',
                    'description': 'Get the flight times between two cities',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'departure': {
                                'type': 'string',
                                'description': 'The departure city (airport code)',
                            },
                            'arrival': {
                                'type': 'string',
                                'description': 'The arrival city (airport code)',
                            },
                        },
                        'required': ['departure', 'arrival'],
                    },
                },
            },
        ],
    )

    messages.append(response['message'])

    if not response['message'].get('tool_calls'):
        print("The model didn't use the function. Its response was:")
        print(response['message']['content'])
        return

    if response['message'].get('tool_calls'):
        available_functions = {
            'get_flight_times': get_flight_times,
        }
        for tool in response['message']['tool_calls']:
            function_to_call = available_functions[tool['function']['name']]
            function_response = function_to_call(tool['function']['arguments']['departure'], tool['function']['arguments']['arrival'])
            messages.append(
                {
                    'role': 'tool',
                    'content': function_response,
                }
            )

    final_response = await client.chat(model=model, messages=messages)
    print(final_response['message']['content'])

asyncio.run(run('mistral'))
```

### Running the Code

To run the code, simply execute the script. It will initialize the conversation, call the `get_flight_times` function as needed, and print the final response from the model.

```bash
python main.py
```
