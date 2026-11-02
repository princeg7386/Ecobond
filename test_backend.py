"""Quick smoke test for the EchoBond backend API."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8080"
PIN = "2468"

def get(path, pin=True):
    req = urllib.request.Request(f"{BASE}{path}")
    if pin:
        req.add_header("X-Caregiver-Pin", PIN)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def post(path, body, pin=True):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if pin:
        req.add_header("X-Caregiver-Pin", PIN)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")

print("=" * 50)
print("EchoBond Backend API Smoke Tests")
print("=" * 50)

# 1. Health
h = get("/api/health", pin=False)
test("/api/health returns ok", h.get("ok") is True)
test("/api/health demo=false (live server)", h.get("demo") is False, f"got {h.get('demo')}")
test("/api/health patient=Kamala", h.get("patient") == "Kamala")

# 2. People
people = get("/api/people")
test("/api/people returns 8 people", len(people) == 8, f"got {len(people)}")
names = [p["name"] for p in people]
test("Anjali Sharma in people", "Anjali Sharma" in names)
test("Ramesh Sharma in people (protected, visible to carer)", "Ramesh Sharma" in names)

# 3. People without PIN (patient view) — only active people
people_patient = get("/api/people", pin=False)
patient_names = [p["name"] for p in people_patient]
test("Patient sees only active people", "Ramesh Sharma" not in patient_names, f"got {patient_names}")

# 4. Memories
mems = get("/api/memories")
test("/api/memories returns 16+", len(mems) >= 16, f"got {len(mems)}")
test("First memory has title", bool(mems[0].get("title")))

# 5. Alerts
alerts = get("/api/alerts")
test("/api/alerts returns 4", len(alerts) == 4, f"got {len(alerts)}")
open_alerts = [a for a in alerts if not a["acknowledged"]]
test("3 open alerts", len(open_alerts) == 3, f"got {len(open_alerts)}")

# 6. Reminders
rems = get("/api/reminders", pin=False)
test("/api/reminders returns data", len(rems) > 0, f"got {len(rems)}")

# 7. Chat
status, chat = post("/api/chat", {"message": "Who is Rohan?", "session": "test"}, pin=False)
test("/api/chat status 200", status == 200, f"got {status}")
test("/api/chat has text", bool(chat.get("text")), f"got {chat}")
test("/api/chat outcome answered", chat.get("outcome") == "answered", f"got {chat.get('outcome')}")
print(f"    Chat response: {chat.get('text', '')[:80]}")

# 8. Chat safety
status, safety = post("/api/chat", {"message": "I'm scared, nobody is here", "session": "test2"}, pin=False)
test("/api/chat safety detection", safety.get("outcome") == "safety", f"got {safety.get('outcome')}")
test("Safety alert level", safety.get("alert") in ("warning", "urgent"), f"got {safety.get('alert')}")

# 9. Context
ctx = get("/api/context", pin=False)
test("/api/context has greeting", "Kamala" in ctx.get("greeting", ""))

# 10. Stats
stats = get("/api/stats")
test("/api/stats has people count", stats.get("people") == 8, f"got {stats.get('people')}")

# 11. Summary
summary = get("/api/summary")
test("/api/summary has lines", len(summary.get("lines", [])) > 0)

# 12. Settings
settings = get("/api/settings", pin=False)
test("/api/settings has patient_name", settings.get("patient_name") == "Kamala")

# 13. Graph
graph = get("/api/graph", pin=False)
test("/api/graph has nodes", len(graph.get("nodes", [])) > 0)
test("/api/graph patient node", graph["nodes"][0].get("patient") is True)

# 14. Places
places = get("/api/places", pin=False)
test("/api/places returns data", len(places) > 0, f"got {len(places)}")

# 15. Interactions
interactions = get("/api/interactions")
test("/api/interactions returns data", len(interactions) > 0, f"got {len(interactions)}")

# 16. Audit
audit_log = get("/api/audit")
test("/api/audit returns data", len(audit_log) > 0)

# 17. Privacy
privacy = get("/api/privacy")
test("/api/privacy has people count", privacy.get("people") == 8)

# 18. Notes
notes = get("/api/notes")
test("/api/notes returns data", len(notes) >= 6, f"got {len(notes)}")

# 19. Frontend served
req = urllib.request.Request(f"{BASE}/")
with urllib.request.urlopen(req) as r:
    html = r.read().decode()[:200]
test("/ serves index.html", "EchoBond" in html)

print("\n" + "=" * 50)
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
print("=" * 50)
