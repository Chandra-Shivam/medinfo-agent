from collections import Counter
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


SECTIONS_TO_INDEX = [
    "indications_and_usage",
    "purpose", 
    "warnings",
    "dosage_and_administration",
    "boxed_warning", 
    "contraindications", 
    "warnings_and_cautions", 
    "drug_interactions", 
    "use_in_specific_populations", 
    "pregnancy", 
    "pediatric_use", 
    "geriatric_use", 
    "overdosage",
    "dosage_forms_and_strengths",
    "spl_medguide"
]



def load_drug_chunks(drug_name:str) -> list[Document]:
    url = "https://api.fda.gov/drug/label.json"
    params = {"search": f'openfda.generic_name:"{drug_name}"', "limit": 1}

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    label = data["results"][0]
    # print(list(label.keys()))
    # print(label["openfda"].get("generic_name"))

    documents = []
    label_type = "Rx" if label.keys() & {"boxed_warning", "warnings_and_cautions"} else "OTC"
    
    for section in SECTIONS_TO_INDEX:
        texts = label.get(section)
        if not texts:
            # print(f"Warning: section '{section}' not found in label")
            continue
        for text in texts:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                    "drug": drug_name,
                    "section": section,
                    "source_id": label["set_id"],
                    "label_type": label_type
                    },
                )
            )

    missing = [s for s in SECTIONS_TO_INDEX if not label.get(s)]
    print(f"{drug_name}: {len(documents)} documents, {len(missing)} candidate sections absent")

    SAFETY_SECTIONS = {"warnings", "warnings_and_cautions", "boxed_warning", "contraindications"}

    indexed_sections = {doc.metadata["section"] for doc in documents}
    if not indexed_sections & SAFETY_SECTIONS:
        raise ValueError(f"{drug_name}: no safety section indexed, refusing to add this drug")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks

DRUGS = ["ibuprofen", "naproxen"]

all_chunks = []

for drug in DRUGS:
    try:
        all_chunks.extend(load_drug_chunks(drug))
    except Exception as e:
        print(f"Skipping {drug} : {e}")

print(Counter(c.metadata["drug"] for c in all_chunks))

emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = Chroma.from_documents(
    documents=all_chunks,
    embedding=emb,
    collection_name="drug_labels",
)
ALLOWED_DRUGS = {chunk.metadata["drug"] for chunk in all_chunks}


