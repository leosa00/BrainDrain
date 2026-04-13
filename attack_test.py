import asyncio
import random
from core.base_attack import TargetConfig, APIFormat
from attacks.reasoning_bomb import ReasoningBombAttack, ReasoningBombConfig, PuzzleLoader
from orchestration.attacker_instance import AttackerInstance, AttackerInstanceConfig
from orchestration.result_collector import ResultCollector
from orchestration.registry import make_config_factory
import secret

google_paid = TargetConfig(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    model="gemini-2.5-flash",
    api_format=APIFormat.CUSTOM,
    api_key=secret.google_api_paid,   # same AIza... key, billing just needs to
    timeout=300.0,                    # be enabled in GCP console for this project
    supports_stream_options=True,
)

runpod_vllm = TargetConfig(
    base_url="https://4bd8ygkrqvok94-8000.proxy.runpod.net",
    model="deepseek-r1-7b",           # must match --served-model-name
    api_format=APIFormat.CUSTOM,       # vLLM is OpenAI-compatible
    api_key=secret.runpod_api_key,
    timeout=1000.0,
)
target = runpod_vllm


N_INSTANCES = 15

PUZZLE_FILE = "prompts/reasoningBomb_puzzles.json"
BUDGET_TIER = "256"

# Pre-select a distinct random puzzle index for each instance so that
# concurrent requests never share a prompt and bypass vLLM's prefix cache.
_loader    = PuzzleLoader(PUZZLE_FILE)
_pool_size = _loader.count(BUDGET_TIER)
_indices   = random.sample(range(_pool_size), min(N_INSTANCES, _pool_size))
# If N_INSTANCES exceeds pool size, wrap around (still shuffled)
while len(_indices) < N_INSTANCES:
    _indices.extend(random.sample(range(_pool_size), min(_pool_size, N_INSTANCES - len(_indices))))

async def smoke_test():
    queue     = asyncio.Queue()
    stop      = asyncio.Event()
    collector = ResultCollector()

    instances = [
        AttackerInstance(AttackerInstanceConfig(
            attack_cls=ReasoningBombAttack,
            attack_config_factory=make_config_factory(target, max_tokens=16000),
            attack_extra_kwargs={"rb_config": ReasoningBombConfig(
                puzzle_file=PUZZLE_FILE,
                budget_tier=BUDGET_TIER,
                puzzle_index=_indices[i],
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