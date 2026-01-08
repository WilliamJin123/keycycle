import asyncio
import os
import sys
from pathlib import Path
from typing import Dict


from dotenv import load_dotenv
from agno.agent import Agent
from key_manager import MultiProviderWrapper 

CURRENT_DIR = Path(__file__).resolve().parent
ENV_PATH = CURRENT_DIR / "local.env"

load_dotenv(dotenv_path=ENV_PATH, override=True)

async def main():
    print(f"--- Loading Env from: {ENV_PATH} ---")
    try:
        cerebras_wrapper = MultiProviderWrapper.from_env(
            provider='cerebras',
            default_model_id='llama-3.3-70b',
            env_file=str(ENV_PATH),
            temperature=0.7
        )
    except Exception as e:
        print(f"Skipping Cerebras: {e}")
        cerebras_wrapper = None

    try:
        groq_wrapper = MultiProviderWrapper.from_env(
            provider='groq',
            default_model_id='llama-3.3-70b-versatile',
            env_file=ENV_PATH,
            top_p=0.95,
        )
    except Exception as e:
        print(f"Skipping Groq: {e}")
        groq_wrapper = None

    try:
        gemini_wrapper = MultiProviderWrapper.from_env(
            provider='gemini',
            default_model_id='gemini-2.5-flash',
            env_file=ENV_PATH,
            top_k=10
        )
    except Exception as e:
        print(f"Skipping Gemini: {e}")
        gemini_wrapper = None

    try:
        openrouter_wrapper = MultiProviderWrapper.from_env(
            provider='openrouter',
            default_model_id='nvidia/nemotron-nano-12b-v2-vl:free',
            env_file=ENV_PATH
        )
    except Exception as e:
        print(f"Skipping OpenRouter: {e}")
        openrouter_wrapper = None

    wrappers: Dict[str, MultiProviderWrapper] = {}
    if cerebras_wrapper: wrappers["Cerebras"] = cerebras_wrapper
    if groq_wrapper: wrappers["Groq"] = groq_wrapper
    if gemini_wrapper: wrappers["Gemini"] = gemini_wrapper
    if openrouter_wrapper: wrappers["OpenRouter"] = openrouter_wrapper

    request_prompt = "Write a 1-sentence interesting fact about space."
    usage_history = {} 

    print(f"\n{'='*20} STARTING REQUESTS {'='*20}")
    print(f"Cloud DB: {os.getenv('TIDB_DB_URL', 'Using Default/None')}")

    for provider_name, wrapper in wrappers.items():
        print(f"\n--- Running {provider_name} ---")
        
        try:
            model = wrapper.get_model(estimated_tokens=500, wait=True, timeout=20)
            
            # 2. Capture Identifier for Reports
            used_key = model.api_key 
            used_model_id = wrapper.default_model_id
            
            usage_history[provider_name] = {
                "key": used_key,
                "model_id": used_model_id
            }
            
            # 3. Initialize Agno Agent with our rotating model
            agent = Agent(model=model, markdown=True)
            
            print(f"-> Using Key: ...{used_key[-8:]}")
            # This triggers ainvoke_stream -> _rotate_credentials -> _get_metrics -> record_usage
            await agent.aprint_response(request_prompt, stream=True, show_reasoning=True)
            
        except Exception as e:
            print(f"-> FAILED: {e}")
            usage_history[provider_name] = None

        # Short sleep to allow the AsyncLogger to process the Turso batch
        await asyncio.sleep(1)

    print(f"\n\n{'='*20} GENERATING REPORTS {'='*20}")

    for provider_name, wrapper in wrappers.items():
        history = usage_history.get(provider_name)
        if not history:
            continue

        print(f"\n\n>>> REPORTS FOR {provider_name.upper()} <<<\n")

        # Global Stats: Now hydrated from Turso in O(1)
        wrapper.print_global_stats()
        
        # Key Stats: Shows current usage + last_429 cooldown status
        wrapper.print_key_stats(identifier=history['key'])

        # Model Stats: Aggregates across your rotation pool
        wrapper.print_model_stats(model_id=history['model_id'])

    # Final cleanup to flush Turso logs before script exit
    for wrapper in wrappers.values():
        wrapper.manager.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass