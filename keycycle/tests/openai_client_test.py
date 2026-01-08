import asyncio
import os
import sys
from pathlib import Path

from keycycle import MultiProviderWrapper

# Adjust path to point to the project root
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = str(PROJECT_ROOT / "local.env")

async def test_provider(provider: str, model_id: str):
    print(f"\n{'='*60}")
    print(f"Testing Provider: {provider.upper()} (Model: {model_id})")
    print(f"{'='*60}")

    try:
        wrapper = MultiProviderWrapper.from_env(
            provider=provider,
            default_model_id=model_id,
            env_file=ENV_PATH
        )
    except Exception as e:
        print(f"Skipping {provider}: Failed to initialize wrapper. Error: {e}")
        return

    # 1. Sync Client
    print(f"\n[{provider}] [Sync] Testing OpenAI Client...")
    try:
        client = wrapper.get_openai_client()
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, what model are you? (Sync)"}],
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"[{provider}] Sync Error: {e}")

    # 2. Async Client
    print(f"\n[{provider}] [Async] Testing Async OpenAI Client...")
    try:
        async_client = wrapper.get_async_openai_client()
        
        response = await async_client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, what model are you? (Async)"}],
        )
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"[{provider}] Async Error: {e}")

    # 3. Stream
    print(f"\n[{provider}] [Stream] Testing Async Stream...")
    try:
        # Re-use async client
        async_client = wrapper.get_async_openai_client()
        stream = await async_client.chat.completions.create(
            messages=[{"role": "user", "content": "Count to 20"}],
            stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
    except Exception as e:
        print(f"[{provider}] Stream Error: {e}")

    # Stats
    print(f"\n[{provider}] --- Usage Stats ---")
    wrapper.print_global_stats()
    
    # Clean up DB threads
    wrapper.manager.stop()


async def main():
    test_cases = [
        # Using a free model for OpenRouter
        {"provider": "openrouter", "model_id": "mistralai/devstral-2512:free"},
        # Common models for Cerebras and Groq
        {"provider": "cerebras", "model_id": "llama3.1-8b"},
        {"provider": "groq", "model_id": "llama-3.1-8b-instant"},
    ]

    for test in test_cases:
        await test_provider(test["provider"], test["model_id"])

if __name__ == "__main__":
    asyncio.run(main())