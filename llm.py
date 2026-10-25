from dotenv import load_dotenv
load_dotenv()

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

# Free tier: 15 requests/minute = 0.25 per second. Stay a bit under it.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,      # 12 per minute
    check_every_n_seconds=0.1,
    max_bucket_size=1,            # no bursts
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    rate_limiter=rate_limiter,
    max_retries=3,                # retry briefly if a 429 still slips through
)