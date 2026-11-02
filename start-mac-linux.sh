#!/usr/bin/env bash
cd "$(dirname "$0")"
echo "Starting EchoBond Backend Server at http://localhost:8080"
echo "Caregiver PIN: 2468"
(sleep 1 && (which xdg-open >/dev/null && xdg-open http://localhost:8080 || which open >/dev/null && open http://localhost:8080)) &
python3 -m backend
