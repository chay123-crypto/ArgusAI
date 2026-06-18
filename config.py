import os

os.makedirs("cache", exist_ok=True)
CACHE_PATH = "cache/cache.db"

CEREBRAS_MODEL="gpt-oss-120b"
GROQ_MODEL="meta-llama/llama-4-scout-17b-16e-instruct"
TEMPERATURE=0.2
MAX_HISTORY=10

OUTPUT_DIR="output"
INPUT_DIR="input"

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ArgusAI"
os.environ["CEREBRAS_API_KEY"] = os.environ.get("CEREBRAS_API_KEY", "your-key-here")
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "your-key-here")  
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY", "your-key-here")
os.environ["TAVILY_API_KEY"]=os.environ.get("TAVILY_API_KEY", "your-key-here")

RETRY_TIMEOUT: int=3
RETRY_MAX_ATTEMPTS: int=3
RETRY_BACKOFF: float=3

dashboard_palettes={
    "clean":{"bg":"#f8fafc","sidebar":"#1e293b","card":"#ffffff","text":"#1e293b","muted":"#64748b","accent":"#6366f1","accent2":"#f43f5e","border":"#e2e8f0","header_text":"#ffffff","plotly":"plotly_white"},
    "latte":{"bg":"#fdf6ec","sidebar":"#3b2a1a","card":"#fffcf7","text":"#3b2a1a","muted":"#92745a","accent":"#c87941","accent2":"#e05c2a","border":"#e8d5b7","header_text":"#ffffff","plotly":"ggplot2"},
    "arctic":{"bg":"#eaf4fb","sidebar":"#023e8a","card":"#ffffff","text":"#023e8a","muted":"#4a90b8","accent":"#0077b6","accent2":"#00b4d8","border":"#b8ddf5","header_text":"#ffffff","plotly":"seaborn"},
    "paper":{"bg":"#fffbeb","sidebar":"#1c1917","card":"#ffffff","text":"#1c1917","muted":"#78716c","accent":"#d97706","accent2":"#b45309","border":"#fde68a","header_text":"#ffffff","plotly":"ggplot2"},
    }