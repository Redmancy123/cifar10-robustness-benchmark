import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime import run_guarded

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--max-hours', type=float, default=24)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    a = parser.parse_args()
    args = a.command[1:] if a.command and a.command[0] == '--' else a.command
    if not args:
        parser.error('a child Python script is required')
    print(run_guarded(args, a.tag, a.max_hours), flush=True)
