from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from config import CEREBRAS_MODEL,GROQ_MODEL,CACHE_PATH,TEMPERATURE
import os

cerebras_llm=ChatCerebras(
    model=CEREBRAS_MODEL,
    cerebras_api_key=os.environ["CEREBRAS_API_KEY"],temperature=TEMPERATURE)

groq_llm=ChatGroq(
    model=GROQ_MODEL,groq_api_key=os.environ["GROQ_API_KEY"],temperature=TEMPERATURE)

llm=cerebras_llm.with_fallbacks([groq_llm])
chat_llm=groq_llm.with_fallbacks([cerebras_llm])

set_llm_cache(SQLiteCache(database_path=CACHE_PATH))