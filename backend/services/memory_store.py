import re
from typing import List, Dict, Any, Optional, Set, Tuple

STOP_WORDS = set(
    "a an the and or of to in on at for with is was were be been it this that you your i me my we our he she they them his her "
    "their who what when where how do did does tell about please know can could remember talk us said much more some any are am".split()
)

NORMALIZE_STOP = set(
    "a an the and or of to in on at for with is was were be been it this that you your i me my we our please tell about do does did can could would will just now today again".split()
)

CONTRACTIONS = {
    "where's": "where is",
    "who's": "who is",
    "what's": "what is",
    "when's": "when is",
    "i'm": "i am"
}

def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9']+", str(text or "").lower())
    return [w for w in words if w not in STOP_WORDS and len(w) > 1]

def normalize_question(q: str) -> str:
    s = q.lower()
    for k, v in CONTRACTIONS.items():
        s = s.replace(k, v)
    words = re.findall(r"[a-z0-9]+", s)
    filtered = sorted(list(set(w for w in words if w not in NORMALIZE_STOP)))
    return " ".join(filtered)

def search_memories(
    query: str,
    memories: List[Dict[str, Any]],
    people_map: Dict[int, Dict[str, Any]],
    places_map: Dict[int, Dict[str, Any]],
    person_ids: List[int] = None,
    allow_private: bool = False,
    exclude_ids: Set[int] = None,
    k: int = 3
) -> List[Dict[str, Any]]:
    if person_ids is None:
        person_ids = []
    if exclude_ids is None:
        exclude_ids = set()

    qt = set(tokenize(query))
    results = []

    for m in memories:
        m_id = m["id"]
        if m_id in exclude_ids:
            continue

        privacy = m.get("privacy", "normal")
        if not allow_private and privacy != "normal":
            continue
        if allow_private and privacy not in ("normal", "private"):
            continue

        # Check if all tagged people are active
        m_ppl_ids = m.get("people", [])
        m_ppl = [people_map.get(pid) for pid in m_ppl_ids if people_map.get(pid)]
        if not all(p.get("status") == "active" for p in m_ppl):
            continue

        # Construct doc representation
        ppl_text = " ".join(f"{p['name']} {p.get('nickname', '')} {p['relationship']}" for p in m_ppl)
        place_text = places_map.get(m.get("place_id", 0), {}).get("name", "")
        doc_text = f"{m.get('title', '')}. {m.get('content', '')} {ppl_text} {place_text} {m.get('year', '')} {m.get('memory_date', '')} {m.get('category', '')} {m.get('song_title', '')}"

        mt = set(tokenize(doc_text))
        tt = set(tokenize(m.get("title", "")))

        overlap = qt.intersection(mt)
        title_hits = qt.intersection(tt)

        if privacy == "private" and not (len(title_hits) > 0 or len(overlap) >= 2):
            continue

        score = (0.55 * len(overlap) + 0.35 * len(title_hits)) / max(len(qt), 1)
        if person_ids and any(pid in m_ppl_ids for pid in person_ids):
            score += 0.5

        if score > 0.08 or (person_ids and any(pid in m_ppl_ids for pid in person_ids)):
            results.append({
                "memory": m,
                "score": round(float(score), 3),
                "overlap": list(overlap)
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]
