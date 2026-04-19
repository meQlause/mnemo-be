import asyncio
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

async def test_mistral():
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    model = os.getenv("MISTRAL_PARSE_MODEL", "mistral-small-latest")
    
    print(f"Testing Mistral with model: {model}")
    print(f"API Key found: {'Yes' if api_key else 'No'}")
    
    if not api_key:
        print("Error: MISTRAL_API_KEY not found in .env")
        return

    llm = ChatMistralAI(
        model=model,
        mistral_api_key=api_key,
        temperature=0
    )
    
    try:
        response = await llm.ainvoke([HumanMessage(content="Say 'Connected'")])
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"🚨 Connection Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_mistral())
