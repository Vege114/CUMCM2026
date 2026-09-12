"""One-shot sequential continuation explicitly approved in this conversation."""

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
status_file = ROOT / 'data/results/exp005/soft-penalty-beta-0.1/status.json'
while not status_file.exists():
    time.sleep(5)
status = json.loads(status_file.read_text())
if status['status'] != 'complete' or status['completed_days'] != 334:
    raise SystemExit('Main run did not finish; do not automatically start the control')
with (ROOT / '.work/exp005/soft-beta-0.01-console.log').open('w') as stream:
    subprocess.run([str(ROOT / '.venv/bin/python'), str(Path(__file__).parent / 'run.py'),
                    '--beta', '0.01', '--out', str(ROOT / 'data/results/exp005/soft-penalty-beta-0.01')],
                   stdout=stream, stderr=subprocess.STDOUT, check=True)
