"""Active Learning: passive review queue + nightly intent-index rebuild.

This block runs at the tail of every request. It logs LLM-assisted and
low-confidence cases to a SQLite review queue, auto-approves the confident ones,
and feeds approved examples back into the semantic intent index. A background
daemon rebuilds and hot-swaps that index nightly at a configurable UTC hour, with
no service restart.
"""
