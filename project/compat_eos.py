"""
JSON action stop tokens: tokenizer EOS + literal `}`.
Separated so Colab can `from project.compat_eos import ...` even on mixed pulls.
"""
from __future__ import annotations

from typing import Any, List


def json_action_eos_token_ids(tokenizer: Any) -> List[int]:
    """
    Token IDs that may end a single JSON action: model EOS plus the last token
    of the string `}`.
    """
    ids: List[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.append(int(eos))
    try:
        rb = tokenizer.encode("}", add_special_tokens=False)
        if rb:
            x = int(rb[-1])
            if x not in ids:
                ids.append(x)
    except Exception:
        pass
    return ids
