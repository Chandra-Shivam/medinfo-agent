import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

url = "https://api.fda.gov/drug/label.json"
params = {"search": 'openfda.generic_name:"ibuprofen"', "limit": 1}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()
data = response.json()

label = data["results"][0]
#print(list(label.keys()))


SECTIONS_TO_INDEX = [
    "indications_and_usage",
    "purpose", 
    "warnings",
    "dosage_and_administration"
]


documents = []

for section in SECTIONS_TO_INDEX:
   texts = label.get(section)
   if not texts:
      print(f"Warning: section '{section}' not found in label")
      continue
   for text in texts:
      documents.append(
         Document(
            page_content=text,
            metadata={
               "drug": "ibuprofen",
               "section": section,
               "source_id": label["set_id"]
               },
         )
      )

print("documents:", len(documents))

for doc in documents:
   print(doc.metadata["section"], len(doc.page_content))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print("chunks:", len(chunks))

for chunk in chunks:
   if chunk.metadata["section"] == "warnings":
      print("Start:", chunk.page_content[:80])
      print("End:", chunk.page_content[-80:])
      print()