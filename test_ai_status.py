import asyncio
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

async def check_mistral_status():
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    model = os.getenv("MISTRAL_PARSE_MODEL", "mistral-small-latest")
    
    if not api_key:
        print("MISTRAL_API_KEY missing")
        return

    llm = ChatMistralAI(
        model=model,
        mistral_api_key=api_key
    )
    
    print(f"Pinging Mistral ({model})...")
    try:
        resp = await llm.ainvoke([HumanMessage(content="Hello")])
        print(f"✅ Mistral is ONLINE. Response: {resp.content[:20]}...")
    except Exception as e:
        print(f"❌ Mistral is OFFLINE: {e}")

if __name__ == "__main__":
    asyncio.run(check_mistral_status())
