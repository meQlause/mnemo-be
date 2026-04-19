import asyncio
import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from app.ai.prompts.note_prompt import extract_event_date_prompt
from app.schemas.note import EventExtraction

async def test_complex_parsing():
    load_dotenv()
    
    # Use the same setup as the main app
    llm = ChatMistralAI(
        model=os.getenv("MISTRAL_PARSE_MODEL", "mistral-small-latest"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0
    )
    
    structured_llm = llm.with_structured_output(EventExtraction)
    chain = extract_event_date_prompt | structured_llm
    
    text = """It was not last spring when the warehouse incident took place — people keep getting that wrong — it was actually the season before, during what felt like the coldest February in years. I had been working at that site for almost three years by then, every weekday without fail, though the Thursday two days before the incident was the last normal shift any of us would remember. A colleague of mine, who had joined the team the first Monday of January that same year, told me afterward that she had sensed something was off "since at least a week before it happened," though I took that with a grain of salt. The investigation report came out roughly six weeks later, and by early the following month the site had been permanently closed — not because of the incident itself, my manager clarified, but due to a licensing issue that had apparently been unresolved since sometime in the third quarter of the previous year."""
    
    ref_date = "2026-04-19"
    
    print(f"Testing text extraction...")
    print(f"Ref Date: {ref_date}")
    
    try:
        result = await chain.ainvoke({"input": text, "reference_date": ref_date})
        print("\n--- AI Result ---")
        print(f"Event Date: {result.event_date}")
        print(f"Confidence: {result.event_confidence}")
        print(f"Reasoning: {result.event_reasoning}")
    except Exception as e:
        print(f"🚨 Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_complex_parsing())
