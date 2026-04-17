# Contributing to Lucebox

Thanks for considering a contribution. Lucebox is a hub of small, self-contained optimization projects. Each one lives under `projects/` with its own README, benchmarks, and code, and the hub stays thin on purpose.

## What we accept

- **Kernel improvements** that preserve correctness and improve `tok/s`, `tok/J`, or memory footprint on the target hardware. Benchmark deltas required.
- **New projects** under `projects/` that fit the thesis: software-defined efficiency for hybrid DeltaNet/Attention LLMs on consumer GPUs. Open an issue first.
- **Benchmark harness work** under `benchmarks/` once that directory starts shipping code.
- **Doc fixes and writeups** — always welcome.

## What we don't accept (yet)

- Support for attention-only architectures (LLaMA, Mistral) — use vLLM or SGLang.
- Multi-tenant batching. Scope is single-user, batch-1 local inference.
- Closed-source dependencies. Everything here has to be reproducible from public sources.

## Before you open a PR

1. **Read the target project's README** and match its style, tone, and measurement methodology.
2. **Benchmark before and after** on the same hardware, at the same power limit, with the same warmup. Numbers without methodology don't get merged.
3. **Run the existing correctness check** (`bench_pp_tg.py` for megakernel) and confirm your change doesn't regress output parity.
4. **One concern per PR.** Kernel changes, docs, and build config go in separate commits or separate PRs.

## Commit message format

Conventional commits:

```
feat(megakernel): fused QKV+RoPE path cuts per-token launch by 1 kernel
fix(dflash): clamp int8 DeltaNet state update before dequant
docs(hub): add DVFS methodology link
```

Allowed types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `bench`, `chore`, `ci`.

## Hardware access

If you want to contribute benchmarks but don't have the hardware:

- We can run numbered runs on our RTX 3090 rig. Open an issue with the PR.
- Apple Silicon numbers need an M-series machine running `powermetrics`, not a remote box.

## Getting help

- [Discord](https://discord.gg/yHfswqZmJQ) — fastest feedback
- [Issues](https://github.com/Luce-Org/lucebox/issues) — for bugs and proposals
- Mention `@Luce-Org/maintainers` on a PR when it's ready for review

## Licensing

By contributing you agree your work is MIT-licensed, same as the rest of the repo.
