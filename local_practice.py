


from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError


messages = [("system", "You are a careful medical-information assistant."),
            ("user", "In two sentences, what is {drug_name} used for?")]


template = ChatPromptTemplate.from_messages(messages)

result = template.invoke({"drug_name": "ibuprofen"})

print(result.to_messages())


class DrugInfo(BaseModel):
    drug: str
    uses:list[str]
    warnings:list[str] = Field(min_length=1)

test_cases = {
    "complete": {
        "drug": "ibuprofen",
        "uses": ["pain relief", "fever reduction"],
        "warnings": ["stomach bleeding", "kidney problems"],
    },
    "warnings missing": {
        "drug": "ibuprofen",
        "uses": ["pain relief", "fever reduction"],
    },
    "warnings empty": {
        "drug": "ibuprofen",
        "uses": ["pain relief", "fever reduction"],
        "warnings": [],
    },
}


for name,data in test_cases.items():
    print(f"--- {name} ---")
    try:
        drug_info = DrugInfo.model_validate(data)
        print(drug_info)
    except ValidationError as e:
        print("Validation error:", e)

    print()