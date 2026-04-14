import os
from agentpro import ReactAgent, create_model
from agentpro.tools import AresInternetTool


def main() -> None:
    openrouter_key = os.getenv("OPEN_ROUTER_KEY")
    ares_key = os.getenv("ARES_API_KEY")

    if not openrouter_key:
        raise EnvironmentError(
            "OPEN_ROUTER_KEY is missing. Please set it in your environment or .env file."
        )
    if not ares_key:
        raise EnvironmentError(
            "ARES_API_KEY is missing. Please set it in your environment or .env file."
        )

    # Create an OpenRouter model client using the OpenRouter API key
    model = create_model(
        provider="openrouter",
        model_name="z-ai/glm-4.5-air:free",
        api_key=openrouter_key,
    )

    # Use the Ares internet search tool for live web queries
    tools = [AresInternetTool(api_key=ares_key)]

    agent = ReactAgent(model=model, tools=tools)

    query = "What is the latest public information about climate change goals for 2030?"
    print("Running query through ReactAgent with OpenRouter + Ares tool...\n")

    response = agent.run(query)

    print("\n=== Agent Response ===")
    print(response.final_answer if hasattr(response, "final_answer") else response)


if __name__ == "__main__":
    main()
