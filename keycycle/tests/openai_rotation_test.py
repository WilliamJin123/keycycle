import asyncio
import os
import sys
from pathlib import Path

from keycycle import MultiProviderWrapper


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = str(PROJECT_ROOT / "local.env")

async def run_client_stress_test(provider: str, model_id: str, limit_attr: str):
    print(f"\n{'='*60}")
    print(f"CLIENT STRESS TEST: {provider.upper()} ({model_id})")
    print(f"Targeting Limit: {limit_attr}")
    print(f"{ '='*60}")

    try:
        wrapper = MultiProviderWrapper.from_env(
            provider=provider,
            default_model_id=model_id,
            env_file=ENV_FILE
        )
    except Exception as e:
        print(f"Skipping {provider}: Failed to initialize wrapper. Error: {e}")
        return

    # 1. Resolve limits to find target loop count
    provider_limits = wrapper.MODEL_LIMITS.get(provider, {})
    limits_config = provider_limits.get(model_id)
    
    # Fallback to default if specific model limit not found
    if not limits_config:
        print(f" Specific limit for '{model_id}' not found. Using 'default'.")
        limits_config = provider_limits.get('default')
    
    if not limits_config:
        print(f" No limits configuration found for {provider}. Cannot stress test.")
        return

    # 2. Get the actual integer value for the limit
    limit_value = getattr(limits_config, limit_attr, None)

    if limit_value is None:
        print(f" Limit '{limit_attr}' is None (unlimited). Cannot stress test.")
        return

    # 3. Set target loops (Limit + 2 to force a switch)
    target_requests = limit_value + 2
    print(f"-> Limit is {limit_value}. Running {target_requests} requests to force rotation.\n")

    # 4. Initialize Async Client
    client = wrapper.get_async_openai_client()

    for i in range(1, target_requests + 1):
        print(f"[{i}/{target_requests}] Requesting...", end="", flush=True)
        try:
            # Fire a cheap request
            # We use max_tokens=5 to keep it fast and cheap
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5
            )
            
            # Note: We can't easily see the key used here because it's abstracted, 
            # but if it succeeds, it means a key was found and used.
            print(" Success")
            
            # Small sleep to avoid burst limits if any (though we want to hit the rate limit)
            # For Gemini 2.5 flash, limit is 5 RPM. 
            # If we send 5 requests in 1 second, we hit it.
            # If we sleep 0.5s, 5 requests take 2.5s. Still well within 1 minute.
            await asyncio.sleep(0.5)

        except RuntimeError as e:
            # Check for our local limiter message
            if "No available keys" in str(e) or "Timeout" in str(e):
                print(f" BLOCKED (Local): {e}")
                print(f"   (This confirms the Local Rate Limiter is working!)")
                break
            else:
                print(f" RuntimeError: {e}")
                break
        except Exception as e:
            # This catches 429s from the provider that slipped past our local limiter
            # or other API errors
            print(f" API Error: {e}")
            await asyncio.sleep(1)

    # Stats
    print(f"\n[{provider}] --- Usage Stats ---")
    wrapper.print_global_stats()
    
    # Clean up DB threads
    wrapper.manager.stop()

async def main():
    # Use gemini as base test case as requested
    # Gemini 2.5 Flash has a limit of 5 RPM in the config
    await run_client_stress_test("gemini", "gemini-2.5-flash", "requests_per_minute")

if __name__ == "__main__":
    asyncio.run(main())
