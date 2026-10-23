import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

messages = [("system", "You are a careful medical-information assistant."),
            ("user", "In two sentences, what is ibuprofen used for?")]

# None = model default; "minimal" is the lowest level Gemini 3 allows (no full "off")
for level in [None, "low", "minimal"]:
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0, thinking_level=level)

    start = time.perf_counter()
    response = llm.invoke(messages)
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata
    reasoning = usage.get("output_token_details", {}).get("reasoning", 0)
    print(f"--- thinking_level={level or 'default'} ---")
    print(f"output_tokens={usage['output_tokens']}  reasoning={reasoning}  "
          f"answer_tokens={usage['output_tokens'] - reasoning}  time={elapsed:.2f}s")
    print(response.text, "\n")
