import asyncio
from core.base_attack import TargetConfig, APIFormat
from attacks.reasoning_bomb import ReasoningBombAttack, ReasoningBombConfig
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

target = google_paid

rb_config = ReasoningBombConfig(
        puzzle_file="prompts/reasoningBomb_puzzles.json",
        budget_tier="256",          # print tokens as they arrive
    )

instance_cfg = AttackerInstanceConfig(
    attack_cls=ReasoningBombAttack,
    attack_config_factory=make_config_factory(target, max_tokens=30000),
    attack_extra_kwargs={"rb_config": rb_config},
)

async def smoke_test():
    queue     = asyncio.Queue()
    stop      = asyncio.Event()
    collector = ResultCollector()

    # Three separate instances firing concurrently
    instances = [AttackerInstance(instance_cfg) for _ in range(3)]

    results = await asyncio.gather(*[inst.run_once() for inst in instances])

    for result in results:
        await queue.put(result)

    stop.set()
    await collector.consume(queue, stop)

    run_result = collector.finalise(instance_stats=[inst.stats for inst in instances])
    print(run_result.summary())

asyncio.run(smoke_test())