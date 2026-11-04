"""Local reward rubrics; official, unchanged prompts live in _vendor.

The multi-reference rubric is a prompt adaptation for canvas-first editing,
not an upstream UnifiedReward training or accuracy claim.
"""

I2I_MULTIREF_PROMPT = """You are a professional digital artist. Evaluate the edited image using the
following rules.

IMPORTANT: Return only one JSON object with "reasoning" (a string) and
"score" (exactly two numbers): [editing success, canvas preservation].
Both scores range from 0 to 25; higher is better.
In "reasoning", use two short sentences starting "Editing:" and "Preservation:"
to explain each assessment with visible evidence and relevant uncertainty.

RULES:

There are {image_count} images in this exact order:
{image_roles}

Image {result_index} is the only result to evaluate; all preceding images are
inputs. Match the instruction's <imageN> labels to the numbered roles above.
Follow the instruction for intended changes; use actual references for
appearance, even if descriptive text conflicts, unless an attribute change
is explicitly requested.

1. Editing success:
Check the requested operation, reference-specific identity or object features,
count, placement, and interaction, then natural integration with the canvas.
The correct category alone is insufficient. Core errors matter more than
minor aesthetic differences. Judge observable details; do not invent unreadable
text or penalize intentionally requested blur.
0 means the edit was not performed, including an unchanged canvas when a
change was requested; 25 means fully successful.

2. Canvas preservation:
Compare with the canvas only where content should remain unchanged. Specific
edits override generic preservation instructions for affected content.
Allow necessary local replacement, hand/arm adjustments, natural occlusion,
and lighting changes. Do not transfer or require preservation of unrelated
donor content. Assess collateral changes independently: editing failure alone
does not lower preservation.
0 means content that should be preserved is extensively changed; 25 means it
is preserved. An unchanged canvas can score highly here even when editing fails.

Editing instruction:
{prompt}
"""


def multiref_edit_prompt(prompt, source_count):
    """Name every image explicitly, keeping dataset <imageN> indices intact."""
    if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 2:
        raise ValueError("Multi-reference editing needs a canvas and at least one donor reference")
    result_index = source_count + 1
    roles = ["- Image 1 (<image1>): the original canvas to be edited."]
    roles.extend(f"- Image {index} (<image{index}>): donor reference supplying the content specified in the instruction." for index in range(2, source_count + 1))
    roles.append(f"- Image {result_index}: the edited result to evaluate.")
    return I2I_MULTIREF_PROMPT.format(
        image_count=result_index,
        image_roles="\n".join(roles),
        source_count=source_count,
        result_index=result_index,
        prompt=prompt,
    )
