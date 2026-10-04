#!/bin/bash
# Run a command while keeping Windows awake (WSL2). Usage: scripts/keep_awake.sh <command> [args...]
#
# A Windows PowerShell helper calls SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
# every 30 s while a flag file exists; the flag is removed when the command ends (also on
# Ctrl-C / error), so the PC can sleep again afterwards. The screen may still switch off (fine).
# NOT prevented: closing the laptop lid (if set to "sleep"), the power button, critical battery.
# Keep this WSL terminal window open while it runs.
set -u
if [ $# -eq 0 ]; then echo "usage: $0 <command> [args...]"; exit 2; fi
flag="$(cd "$(dirname "$0")/.." && pwd)/logs/.keep_awake_$$"
mkdir -p "$(dirname "$flag")" && touch "$flag"
winflag="$(wslpath -w "$flag")"
cleanup() { rm -f "$flag"; }
trap cleanup EXIT INT TERM
powershell.exe -NoProfile -NonInteractive -Command "
\$t = Add-Type -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);' -Name P -Namespace W -PassThru
while (Test-Path -LiteralPath '$winflag') { [void]\$t::SetThreadExecutionState([uint32]2147483649); Start-Sleep -Seconds 30 }
[void]\$t::SetThreadExecutionState([uint32]2147483648)
" > /dev/null 2>&1 &
ps_pid=$!
echo "[keep_awake] Windows sleep blocked while running: $*  ($(date))"
"$@"
status=$?
cleanup
wait "$ps_pid" 2>/dev/null
echo "[keep_awake] finished with exit code $status, sleep allowed again ($(date))"
exit $status
