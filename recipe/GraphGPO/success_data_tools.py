"""Outcome-verified action supervision; no reward or preference relabeling.

Mask by decoding the stored token prefixes, not by retokenizing text. Qwen
generation can produce alternative tokenizations of identical visible text.
"""

import collections
import random
import re


def action_mask(tokenizer, ids, expected_action):
    prefixes = [""] + [
        tokenizer.decode(ids[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for i in range(1, len(ids) + 1)
    ]
    text = prefixes[-1]
    m = re.search(r"<action>\s*(.*?)\s*</action>", text, re.S | re.I)
    assert m is not None, "Missing action block"
    # The production projector executes the FIRST block. Later action text
    # must never become a target for the action-specific loss.
    assert m.group(1).strip().lower() == expected_action.strip().lower()
    start, end = m.span(1)
    mask = []
    for i in range(len(ids)):
        before, after = prefixes[i], prefixes[i + 1]
        overlap = len(after) > start and len(before) < end
        if overlap:
            # Action strings are ASCII; prohibit ambiguous Unicode offsets or
            # a token prefix that does not correspond to the complete decode.
            assert text.startswith(before) and text.startswith(after)
            assert len(after) > len(before)
        mask.append(int(overlap))
    assert any(mask)
    chosen = [i for i, bit in enumerate(mask) if bit]
    assert chosen == list(range(min(chosen), max(chosen) + 1))
    assert len(prefixes[min(chosen)]) <= start
    assert len(prefixes[max(chosen) + 1]) >= end
    return mask


def trim_unexecuted_tail(tokenizer, ids, expected_action):
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    m = re.search(r"<action>\s*(.*?)\s*</action>", text, re.S | re.I)
    assert m is not None and m.group(1).strip().lower() == expected_action.strip().lower()
    think_end = text.find("</think>")
    assert think_end >= 0, "Missing required reasoning closure"
    end = max(m.end(), think_end + len("</think>"))
    assert (
        len(re.findall(r"<action>", text[:end], re.I)) == 1
    ), "Cannot trim nested multiple actions safely"
    if not text[end:].strip():
        return list(ids), False
    # Preserve the original tokenization wherever the closing tag ends at a
    # token boundary. Only retokenize if a token also contains later prose.
    for n in range(1, len(ids) + 1):
        prefix = tokenizer.decode(
            ids[:n], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if len(prefix) >= end:
            if prefix.startswith(text[:end]) and not prefix[end:].strip():
                kept = list(ids[:n])
            else:
                kept = tokenizer.encode(text[:end], add_special_tokens=False)
            assert kept[-1] != tokenizer.eos_token_id
            return kept + [tokenizer.eos_token_id], True
    raise AssertionError("Could not locate executed action boundary")


class BalancedCycle:
    def __init__(self, values, rng):
        assert len(values) == len(set(values)) and values
        self.values, self.rng, self.pending = list(values), rng, []

    def draw(self):
        if not self.pending:
            self.pending = self.values.copy()
            self.rng.shuffle(self.pending)
        return self.pending.pop()


def make_orders(rows, groups, updates, seed):
    rng = random.Random(seed)
    critical = [i for i, r in enumerate(rows) if r["kind"] == "repair" and r["intervention"]]
    assert len(critical) == len(groups["repair"])
    assert len({rows[i]["goal_index"] for i in critical}) == len(critical)
    critical_cycle = BalancedCycle(critical, rng)
    normal_cycle = BalancedCycle(sorted(groups["normal"]), rng)

    def normal():
        goal = normal_cycle.draw()
        return rng.choice(rng.choice(groups["normal"][goal]))

    orders = []
    for u in range(1, updates + 1):
        natural = [normal() for _ in range(8)]
        keys = [critical_cycle.draw() for _ in range(2)]
        orders.append(
            dict(
                update=u,
                normal=natural,
                repair=natural[:6] + keys,
                reference=[normal() for _ in range(2)],
            )
        )
    counts = collections.Counter(i for o in orders for i in o["repair"] if rows[i]["intervention"])
    assert set(counts) == set(critical)
    assert max(counts.values()) - min(counts.values()) <= 1
    return orders, {str(rows[i]["goal_index"]): counts[i] for i in critical}
