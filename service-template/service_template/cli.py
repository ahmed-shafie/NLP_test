"""Headless REPL — talk to the engine straight from the terminal (no server).

Run it with::

    python -m service_template.cli

Type messages; the assistant replies. When a dialogue completes, the emitted
JSON action object is printed. Type ``quit`` to exit. This is the fastest way to
sanity-check a new case while you build it.
"""

from __future__ import annotations

import uuid

from service_template.engine import get_engine


def main() -> None:
    engine = get_engine()
    session_id = uuid.uuid4().hex
    print("Service Template REPL — type 'quit' to exit.\n")
    print("Try: send 500 SAR to Ahmed\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break
        result = engine.handle(text, session_id)
        print(f"bot> {result.reply}")
        if result.action is not None:
            # The structured JSON action object for a downstream system.
            print("\n--- ACTION OBJECT (JSON) ---")
            print(result.action.model_dump_json(indent=2))
            print("---------------------------\n")


if __name__ == "__main__":
    main()
