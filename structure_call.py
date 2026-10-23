from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


load_dotenv()

class DrugInfo(BaseModel):
    drug : str
    uses: list[str]
    warnings: list[str] = Field(min_length=1)


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

structured = llm.with_structured_output(DrugInfo)

try:
    result = structured.invoke("Summarize ibuprofen")
    print("Type returned:", type(result))
    print("Drug:", result.drug)
    print("Uses:", result.uses)
    print("Warnings:", result.warnings)
except Exception as e:
    print("Something went wrong:", e)