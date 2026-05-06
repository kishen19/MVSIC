#!/usr/bin/env python3
"""Batch-stage breakdown intentionally disabled.

Batch runs currently do not return the substage timing columns needed for the
MV-IVF breakdown plot, so this wrapper exits successfully without writing a
PDF.
"""
from __future__ import annotations

def main() -> int:
    print("[batch breakdown] skipped: batch runs do not emit substage timings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
