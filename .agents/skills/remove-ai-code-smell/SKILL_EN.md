---
name: remove-ai-code-smell
description: Review and simplify code that looks overengineered or AI-generated. Remove excessive defensive logic in internal configuration paths, exhaustive handling of impossible input formats, unjustified capability guards, redundant exceptions, and boilerplate comments while preserving algorithmic and domain checks that prevent silent failures. Use when asked to remove AI code smell, simplify code, eliminate over-defensive checks, rely on project contracts instead of treating trusted inputs as hostile, or audit suspicious guards, validations, and comments.
---

# Remove AI Code Smells

## Core Principles

Trust verified caller contracts. Do not treat internal project configuration as an untrusted public API, and do not enumerate formats that callers never provide.

First distinguish between two kinds of checks:

- **Format defenses**: Handling `None`, capitalization variants, `bool`, arbitrary invalid types, or configuration combinations the project never produces. Remove these checks; when necessary, state the input contract in one sentence.
- **Real boundaries**: Preventing division by zero, out-of-bounds access, duplicate indices, silent selection of the wrong strategy, data corruption, or resource safety issues. Preserve these checks.

Do not keep every guard merely because it appears safer. Determine what happens when it fails. If it only makes an internal error surface earlier and more verbosely, it is usually unnecessary. If execution would continue and silently produce an incorrect result, keep it.

In LightX2V, startup JSON and request data are separate contracts. Inspect `build_cli_inputs()` / `build_startup_config()` in `lightx2v/utils/set_config.py` and `BaseRunner.prepare_request()` in `lightx2v/models/runners/base_runner.py`. Task/field validation at this public boundary is not redundant internal defense.

## Review Workflow

1. Inspect call sites, configuration files, and defaults to determine who controls each input.
2. Find `isinstance` chains, `None` fallbacks, case normalization, repeated `hasattr` checks, capability-combination guards, and long exception messages.
3. Check that optional configuration reads such as `config.get` provide an explicit, appropriate default consistent with the setting's semantics, caller contract, and existing behavior when omitted. Use `config[key]` for required settings; do not hide missing settings with arbitrary defaults or use `or default` to replace valid `False`, `0`, or other falsy values.
4. Remove branches outside the actual input space so the code directly expresses the normal path.
5. For stable but non-obvious contracts, keep at most one short comment. Do not restate the code in comments.
6. Preserve the minimum boundary checks needed to prevent silent errors.
7. Keep changes local, then run the relevant syntax checks, lint checks, and focused tests.

## Common Simplifications

Replace exhaustive handling of internal configuration formats:

```python
if value is None:
    value = 0
if isinstance(value, str):
    if value.lower() != "all":
        raise ValueError(...)
    count = num_blocks
elif isinstance(value, bool) or not isinstance(value, int):
    raise ValueError(...)
else:
    count = value
```

with a clear contract and the normal path:

```python
# Resident block counts are integers or "all".
count = num_blocks if value == "all" else value
```

An untested configuration combination is not proof that callers cannot select it. Keep explicit unsupported-mode guards when they protect computation or resource lifetimes.

## Checks to Preserve

Keep boundaries required by the algorithm itself. For example, a resident block count greater than the total number of layers can produce duplicate indices or an invalid layout, so the range check is not excessive:

```python
if not 0 <= count <= num_blocks:
    raise ValueError(...)
```

Also preserve:

- Validation at public API boundaries and for user input, network data, and untrusted files.
- Safety boundaries around distributed collectives, VRAM capacity, file overwrites, and data persistence.
- Strategy or enum checks where invalid input would otherwise be accepted and produce incorrect results.

Do not remove defensive checks mechanically. Prove the caller contract first, then simplify the code.
