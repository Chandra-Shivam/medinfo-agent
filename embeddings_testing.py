import math
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

texts = [
    "You can take this with food.",     # 0
    "You cannot take this with food.",  # 1
    "Kidney disease",                   # 2
    "Renal impairment",                 # 3
    "Chocolate cake recipe",            # 4
]

vectors = emb.embed_documents(texts)
print("vectors:", len(vectors), "| numbers per vector:", len(vectors[0]))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

print(cosine(vectors[0], vectors[1]))  # expect negative
print(cosine(vectors[2], vectors[3]))  # expect positive
print(cosine(vectors[2], vectors[4]))  # expect minimum similarity