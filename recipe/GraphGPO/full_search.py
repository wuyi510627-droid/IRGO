import json, re


def parse_search_proposals(raw, previous):
    """One proposal request only; invalid/duplicate outputs are not regenerated."""
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        obj = json.loads(text)
    except (ValueError, TypeError):
        return []
    if not isinstance(obj, list):
        return []
    result = []
    for query in obj[:3]:
        if not isinstance(query, str):
            continue
        query = " ".join(query.split()).lower()
        action = f"search[{query}]"
        if (
            not 2 <= len(query) <= 180
            or "[" in query
            or "]" in query
            or re.search(r"[\u4e00-\u9fff]", query)
            or action in previous
            or action in result
        ):
            continue
        result.append(action)
    return result[:2]
