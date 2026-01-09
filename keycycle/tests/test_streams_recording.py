import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from keycycle import MultiProviderWrapper

ENV_PATH = current_dir / "local.env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

async def test_provider(provider: str, model_id: str):
    print(f"\n{'='*60}")
    print(f"Testing Provider: {provider.upper()} ({model_id})")
    print(f"{'='*60}")

    try:
        wrapper = MultiProviderWrapper.from_env(
            provider=provider,
            default_model_id=model_id,
            env_file=str(ENV_PATH)
        )
    except Exception as e:
        print(f"SKIPPING {provider}: {e}")
        return

    # Initial stats
    initial_stats = wrapper.manager.get_global_stats()
    initial_reqs = initial_stats.total.total_requests
    initial_tokens = initial_stats.total.total_tokens
    print(f"Initial Requests: {initial_reqs}, Initial Tokens: {initial_tokens}")

    # 1. Sync Stream
    print("\n--- Sync Stream ---")
    try:
        client = wrapper.get_openai_client()
        # Note: 'stream_options' with 'include_usage' is critical for usage reporting in streaming
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": "Write a one word poem."}
            ],
            max_tokens=10,
            stream=True,
            stream_options={"include_usage": True} 
        )
        content = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        print(f"Received: {content.strip()}")
    except Exception as e:
        print(f"Sync Stream FAILED: {e}")

    # 2. Async Stream
    print("\n--- Async Stream ---")
    try:
        aclient = wrapper.get_async_openai_client()
        # The wrapper should automatically inject stream_options={"include_usage": True} for async
        stream = await aclient.chat.completions.create(
            messages=[{"role": "user", "content": "Write another one word poem."}
            ],
            max_tokens=10,
            stream=True
        )
        content = ""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        print(f"Received: {content.strip()}")
    except Exception as e:
        print(f"Async Stream FAILED: {e}")

    # 3. Verify Usage
    # Give a small buffer if there's any background processing (though stats should be immediate)
    await asyncio.sleep(0.5)
    
    final_stats = wrapper.manager.get_global_stats()
    final_reqs = final_stats.total.total_requests
    final_tokens = final_stats.total.total_tokens

    print(f"\nFinal Requests: {final_reqs}, Final Tokens: {final_tokens}")
    
    req_diff = final_reqs - initial_reqs
    token_diff = final_tokens - initial_tokens
    
    print(f"Delta Requests: {req_diff} (Expected >= 2)")
    print(f"Delta Tokens: {token_diff} (Expected > 0)")

    success = True
    if req_diff < 2:
        print(f"❌ FAIL: Requests not recorded properly (Got {req_diff}, Expected 2)")
        success = False
    
    # OpenRouter free models sometimes return 0 usage, so we warn instead of fail for them if tokens are 0
    # But generally we expect tokens.
    if token_diff <= 0:
        if "free" in model_id.lower() or "nano" in model_id.lower():
             print(f"⚠️  WARN: Tokens not recorded (Got {token_diff}). This might be typical for some free models.")
        else:
            print(f"❌ FAIL: Tokens not recorded properly (Got {token_diff})")
            success = False
            
    if success:
        print(f"✅ {provider.upper()} PASSED Usage Check")

    wrapper.manager.stop()

async def main():
    test_cases = [
        {"provider": "cerebras", "model_id": "llama3.1-8b"},
        {"provider": "groq", "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct"},
        {"provider": "openrouter", "model_id": "xiaomi/mimo-v2-flash:free"},
    ]

    for case in test_cases:
        await test_provider(case["provider"], case["model_id"])

if __name__ == "__main__":
    asyncio.run(main())
