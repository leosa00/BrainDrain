import asyncio
from core.base_attack import AttackConfig, TargetConfig, APIFormat
from attacks.reasoning_bomb import ReasoningBombAttack, ReasoningBombConfig
import secret

google = TargetConfig(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    model="gemini-2.5-flash",
    api_format=APIFormat.CUSTOM,
    api_key=secret.google_api_paid, 
)

local = TargetConfig(
        base_url="http://localhost:11434",
        model="deepseek-r1:8b",       # has thinking/reasoning mode
        api_format=APIFormat.CUSTOM,
        api_key=None,
        timeout=600.0,
    )

runpod_vllm = TargetConfig(
    base_url="https://wxjz5hyynv2a6n-8000.proxy.runpod.net",
    model="deepseek-r1-7b",           # must match --served-model-name
    api_format=APIFormat.CUSTOM,       # vLLM is OpenAI-compatible
    api_key=secret.runpod_api_key,
    timeout=600.0,
)

async def main():
    target = runpod_vllm

    config = AttackConfig(
        target=target,
        max_tokens=10000,
        stream=True,
    )
    rb_cfg = ReasoningBombConfig(
        puzzle_file="prompts/reasoningBomb_puzzles.json",
        budget_tier="256",
        verbose_stream=True,            # print tokens as they arrive
    )

    attack = ReasoningBombAttack(config, rb_cfg)
    result = await attack.run()
    await attack.close()

    print("\n\n─── Result ───")
    for k, v in result.to_dict().items():
        print(f"{k:30s}: {v}")

asyncio.run(main())
