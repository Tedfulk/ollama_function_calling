import json
from unittest.mock import AsyncMock, patch

import pytest

from ollama_function_calling.main import get_flight_times, run


def test_get_flight_times():
    # Test valid flight
    result = get_flight_times("NYC", "LAX")
    expected = {
        "departure": "08:00 AM",
        "arrival": "11:30 AM",
        "duration": "5h 30m",
    }
    assert json.loads(result) == expected

    # Test invalid flight
    result = get_flight_times("ABC", "XYZ")
    assert json.loads(result) == {"error": "Flight not found"}


@pytest.mark.asyncio
async def test_run():
    mock_client = AsyncMock()
    mock_client.chat.side_effect = [
        {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_flight_times",
                            "arguments": {"departure": "NYC", "arrival": "LAX"},
                        }
                    }
                ]
            }
        },
        {"message": {"content": "The flight from NYC to LAX takes 5h 30m."}},
    ]

    with patch(
        "ollama_function_calling.main.ollama.AsyncClient", return_value=mock_client
    ):
        await run("mistral")

    assert mock_client.chat.call_count == 2
    mock_client.chat.assert_called_with(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": "What is the flight time from New York (NYC) to Los Angeles (LAX)?",
            },
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_flight_times",
                            "arguments": {"departure": "NYC", "arrival": "LAX"},
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "content": '{"departure": "08:00 AM", "arrival": "11:30 AM", "duration": "5h 30m"}',
            },
        ],
    )


if __name__ == "__main__":
    pytest.main()
