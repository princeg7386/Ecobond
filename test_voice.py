"""Test the voice cloning endpoints."""
import json
import urllib.request
import urllib.error
import os

BASE = "http://localhost:8080"
PIN = "2468"

def post_json(path, body, pin=True):
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if pin:
        req.add_header("X-Caregiver-Pin", PIN)
    try:
        with urllib.request.urlopen(req) as r:
            ct = r.headers.get("Content-Type", "")
            if "audio" in ct:
                audio = r.read()
                return r.status, {"audio_size": len(audio), "content_type": ct}
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

def get(path, pin=True):
    req = urllib.request.Request(f"{BASE}{path}")
    if pin:
        req.add_header("X-Caregiver-Pin", PIN)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

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
print("Voice Cloning API Tests")
print("=" * 50)

# 1. /api/speak with no person_id (system voice)
status, data = post_json("/api/speak", {"text": "Good morning Kamala"}, pin=False)
test("/api/speak system voice", status == 200 and data.get("audio_size", 0) > 1000, f"status={status} data={data}")

# 2. /api/speak with person_id (no voice sample yet for most)
status, data = post_json("/api/speak", {"text": "Hello Kamala", "person_id": 1}, pin=False)
test("/api/speak person without voice -> system fallback", status in (200, 204), f"status={status}")

# 3. /api/voices
voices = get("/api/voices", pin=False)
test("/api/voices returns list", isinstance(voices, list), f"got {type(voices)}")

# 4. /api/people/1/voice GET
status2, data2 = 200, get("/api/people/1/voice")
test("/api/people/1/voice GET", data2.get("person_id") == 1, f"got {data2}")

# 5. /api/voice-chat (ask about Rohan)
status, data = post_json("/api/voice-chat", {"message": "Where is Rohan?", "session": "vtest"}, pin=False)
test("/api/voice-chat returns text", bool(data.get("text")), f"got {data}")
test("/api/voice-chat has segments", len(data.get("segments", [])) > 0, f"got {data.get('segments')}")
print(f"    Voice chat: {data.get('text', '')[:80]}")
print(f"    Mode: {data.get('mode')}, Engine: {data.get('engine')}")

# 6. /api/voice-chat (talk to Priya)
status, data = post_json("/api/voice-chat", {"message": "I want to talk to Priya", "session": "vtest2"}, pin=False)
test("/api/voice-chat talk request", "talk" in data.get("text", "").lower() or "priya" in data.get("text", "").lower(), f"got {data.get('text', '')[:80]}")

# 7. /api/voice-chat safety (falls back to normal)
status, data = post_json("/api/voice-chat", {"message": "I'm scared nobody is here", "session": "vtest3"}, pin=False)
test("/api/voice-chat safety fallback", data.get("mode") == "echobond", f"got mode={data.get('mode')}")

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'='*50}")
