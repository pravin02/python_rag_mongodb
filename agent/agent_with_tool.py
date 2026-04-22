from datetime import datetime

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.capabilities import Thinking, WebSearch


model = OllamaModel(
    "gemma4", provider=OllamaProvider(base_url="http://localhost:11434/v1")
)


agent = Agent(
    model,
    instructions="Be concise, reply with one sentence.",
    capabilities=[Thinking(), WebSearch()],
)


# @agent.tool_plain
# def current_datetime() -> datetime:
#     """This tool return the users current date time"""
#     print("current_datetime: ", datetime.now())
#     return datetime.now()


while True:
    query = input("User: ")
    if query == "quite" or query == "exit":
        print("User prefered to exit from program")
        break
    result = agent.run_sync(query)
    print("Agent: ", result.output)
