import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / "local.env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.cerebras import Cerebras
from agno.models.google.gemini import Gemini
from agno.models.openrouter import OpenRouter
from agno.utils.pprint import pprint_run_response

def test_provider_keys(prefix, 
    provider_class, 
    model_id, 
    prompt="Say Hello.",
    start = 1,
    end = None
):
    num = int(os.getenv(f"NUM_{prefix}", 0))
    success = []
    fail = []
    end = end if end is not None else num + 1
    for i in range(start, end):
        key_env = f"{prefix}_API_KEY_{i}"
        print("CALLING WITH KEY:", key_env)
        key = os.getenv(key_env)
        if not key:
            fail.append(key_env)
            continue
        try:
            agent = Agent(model=provider_class(id=model_id, api_key=key), markdown=True)
            r = agent.run(prompt)
            pprint_run_response(r)
            if "429" in str(r) or "quota" in str(r).lower():
                raise Exception("Rate limit or quota exceeded")
            success.append(key_env)
        except Exception:
            fail.append(key_env)
    print(f"{prefix} successful:", success)
    print(f"{prefix} failed:", fail)
    input("Press Enter to continue...")

# test_provider_keys("GROQ", Groq, "llama-3.3-70b-versatile")
# test_provider_keys("CEREBRAS", Cerebras, "llama-3.3-70b")
# test_provider_keys("GEMINI", Gemini, "gemini-2.5-flash")
test_provider_keys("OPENROUTER", OpenRouter, "xiaomi/mimo-v2-flash:free", start=26)

