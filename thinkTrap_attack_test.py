import asyncio
import random
from core.base_attack import TargetConfig, APIFormat
from attacks.think_trap import ThinkTrapAttack, ThinkTrapConfig, ThinkTrapPromptCache
from orchestration.attacker_instance import AttackerInstance, AttackerInstanceConfig
from orchestration.result_collector import ResultCollector
from orchestration.registry import make_config_factory
import secret

google_paid = TargetConfig(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    model="gemini-2.5-flash",
    api_format=APIFormat.CUSTOM,
    api_key=secret.google_api_paid,
    timeout=300.0,
    supports_stream_options=True,
)

runpod_vllm = TargetConfig(
    base_url="https://wxjz5hyynv2a6n-8000.proxy.runpod.net",
    model="deepseek-r1-7b",           # must match --served-model-name
    api_format=APIFormat.CUSTOM,       # vLLM is OpenAI-compatible
    api_key=secret.runpod_api_key,
    timeout=600.0,
)
target = runpod_vllm

N_INSTANCES = 1

# Must match a file produced by thinkTrap_test.py.  Choose one of: 20, 40, 60, 80.
P_LENGTH    = 20
PROMPT_FILE = f"prompts/thinktrap_prompts_{P_LENGTH}.json"

# Pre-select distinct random prompt indices so concurrent requests never share
# a prompt and bypass vLLM's prefix cache.
_cache     = ThinkTrapPromptCache(PROMPT_FILE)
_pool_size = _cache.count()
_indices   = random.sample(range(_pool_size), min(N_INSTANCES, _pool_size))
while len(_indices) < N_INSTANCES:
    _indices.extend(random.sample(range(_pool_size), min(_pool_size, N_INSTANCES - len(_indices))))


async def smoke_test():
    queue     = asyncio.Queue()
    stop      = asyncio.Event()
    collector = ResultCollector()

    print(f"\n=== Prompts being sent (from {PROMPT_FILE}) ===")
    for i, idx in enumerate(_indices):
        entry = _cache.get(idx)
        print(f"  [{i}] index={idx}  score={entry.get('score')}  text={entry.get('text', '')!r}")
    print("=" * 52 + "\n")

    instances = [
        AttackerInstance(AttackerInstanceConfig(
            attack_cls=ThinkTrapAttack,
            attack_config_factory=make_config_factory(target, max_tokens=16000),
            attack_extra_kwargs={"tt_config": ThinkTrapConfig(
                prompts_file=PROMPT_FILE,
                prompt_index=_indices[i],
            )},
        ))
        for i in range(N_INSTANCES)
    ]

    results = await asyncio.gather(*[inst.run_once() for inst in instances])

    for result in results:
        await queue.put(result)

    stop.set()
    await collector.consume(queue, stop)

    run_result = collector.finalise(instance_stats=[inst.stats for inst in instances])
    print(run_result.summary())


asyncio.run(smoke_test())
