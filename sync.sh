#!/bin/bash
git add -A
git commit -m "Auto-sync: $(date '+%Y-%m-%d %H:%M:%S')"
git push origin main
echo "Successfully synced with GitHub!"
