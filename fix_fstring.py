import sys

with open('backend/services/conversation.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Replace lines 344-351 (1-indexed) = 343-350 (0-indexed)
new_block = [
    '        when = top_mem.get("memory_date") or top_mem.get("year")\n',
    '        m_ppl = [people_map[pid] for pid in top_mem.get("people", []) if pid in people_map]\n',
    '        m_names = [display_name(p["name"]) for p in m_ppl]\n',
    '        when_part = f" ({when})" if when else ""\n',
    '        names_joined = ", ".join(m_names)\n',
    '        mentions_part = f" mentions {names_joined}" if m_names else ""\n',
    '        why_items.append({\n',
    '            "type": "memory",\n',
    '            "memory_id": top_mem["id"],\n',
    '            "text": "\\u201c" + top_mem["title"] + "\\u201d" + when_part + mentions_part\n',
    '        })\n',
]

lines[343:351] = new_block

with open('backend/services/conversation.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Fixed conversation.py line 344-351")
