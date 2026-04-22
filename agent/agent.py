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

result = agent.run_sync("What was the mass of the largest meteorite found this year?")
print(result.output)
