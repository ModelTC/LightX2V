# Ordinary runner boundary expectations

Static CPU observations for 86 explicitly catalogued boundaries on upstream
main 43c6a11b. Upstream removed or refactored several internal-fork runners;
the catalogue follows current methods and shared autoregressive lifecycles.
Tests do not regenerate these expectations.

Each row pins exception/return outcome, ordered cleanup events, actual mock calls
and released reference state. Typed stand-ins preserve causal-cache and video
probe return contracts. Production changes were compared with upstream after
normalizing only fatal-check handlers/guards and unused exception bindings;
ordinary operations and cleanup structure are retained.

Mutation checks reject missing/throwing cleanup, changed exception/return values,
late process exit and fallback calls preceding fatal termination. Child crashes
and missing/malformed observations cannot count as detected behavior changes.
Separate full-method tests cover capture, GPU-copy streams and writer threads.
These are Python control-flow checks, not GPU hardware/recovery simulations.
