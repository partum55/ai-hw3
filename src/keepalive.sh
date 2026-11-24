#!/bin/sh
# Keeps the friend's WSL2 alive: one SSH session stays open and its remote loop writes a heartbeat every 15 min
# (WSL stops when nothing is running in an attached session). Reconnects if the link drops. Stop: kill $(cat preds/keepalive.pid)
HOST=sasha@100.86.126.37
LOG="$(dirname "$0")/../preds/keepalive.log"
while true; do
  ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "$HOST" \
    'mkdir -p /home/ubuntu/hw3/logs; while true; do date -Is >> /home/ubuntu/hw3/logs/keepalive.log; sleep 900; done'
  echo "$(date -Is) ssh dropped, reconnecting in 15 s" >> "$LOG"; sleep 15
done
