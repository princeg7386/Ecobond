# EchoBond (अपनों की आवाज़) 🌿

> **An AI Memory, Facial Recognition & Voice Companion Assistant for Alzheimer’s & Dementia Patients**

EchoBond is designed to bring gentle reassurance, cognitive stimulation, and comfort to individuals living with Alzheimer’s or dementia. It bridges the gap between caregivers and loved ones through memory recall, consented face recognition, daily orientation, and a realistic **cloned voice companion** that allows patients to talk to their family members in their own familiar voices even when they are away.

---

## ✨ Key Features

### 🎙️ Apno Ki Awaaz (Family Voice Companion)
- **Consented Voice Cloning & Pitch Tuning**: Uses Neural TTS calibrated with sample audio to approximate family members' pitch and speaking pace.
- **Dynamic 1st-Person Familial Persona**: Rather than robotic third-person responses, family members (e.g. Rohan, Priya, Anjali) converse warmly in the first person ("Main", "Dadi", "Nani", "Ma").
- **Time & Schedule Aware**: Responds accurately about whereabouts based on real-time caregiver schedules and reminders.
- **Shared Memory Recall**: Retells real, verified family memories and past events in their own voice.
- **Google Gemini Generative AI Integration**: Optional Gemini LLM integration for open-ended, empathetic conversation grounded in patient facts and safety guardrails.

### 👁️ Consented Face Recognition (Who is this?)
- Biometric verification with strict lighting, blur, and margin checks to prevent misidentifications.
- Caregiver-managed consent controls and photo training.

### 🛡️ Privacy, Safety & Guardrails
- **Distress Detection**: Automatically detects confusion, fear, or fall alarms, providing gentle grounding and alerting caregivers immediately.
- **Protected Persons**: Safely handles deceased or sensitive family members without causing trauma or confusion.
- **Audit Logs**: Transparent, verifiable activity logs with zero unauthorized data sharing.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.9+
- Pip & Git

### 2. Installation
```bash
git clone https://github.com/princeg7386/Ecobond.git
cd Ecobond
pip install -r requirements.txt
```

### 3. Run EchoBond
- **On Windows:** Double-click `start-windows.bat` or run:
  ```bash
  python -m backend
  ```
- **On Mac/Linux:**
  ```bash
  chmod +x start-mac-linux.sh
  ./start-mac-linux.sh
  ```

Open your browser at **`http://localhost:8080`**.

Default Caregiver PIN: `2468`

---

## 📁 Project Structure

```
├── assets/                 # Icons, fonts, and family member avatars
├── backend/
│   ├── app.py              # Flask API server & endpoint routes
│   ├── config.py           # Environment & directory configurations
│   ├── database.py         # SQLite schema, migrations & helpers
│   ├── data/               # SQLite database (echobond.db)
│   ├── services/
│   │   ├── conversation.py # Memory retrieval & conversational engine
│   │   ├── face_service.py # Quality checks & face embeddings
│   │   ├── persona_service.py # Dynamic 1st-person family dialogue & Gemini AI
│   │   ├── safety.py       # Distress alerts, clinical safeguards & grounding
│   │   └── voice_service.py# Pitch extraction & Edge Neural TTS synthesis
├── index.html              # Frontend Single-Page Application (Patient & Caregiver UI)
├── requirements.txt        # Python dependencies
├── start-windows.bat       # Windows one-click startup script
└── start-mac-linux.sh      # Unix startup script
```

---

## 🔒 Privacy & Clinical Ethics
EchoBond adheres to strict eldercare design principles:
- No medical advice or dosage modifications without caregiver oversight.
- Never fabricates unknown individuals or false memories.
- Real-time caregiver alerts for wandering, distress, or repeated agitation.
