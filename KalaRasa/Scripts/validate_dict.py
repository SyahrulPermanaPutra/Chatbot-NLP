#!/usr/bin/env python3
"""
scripts/validate_dict.py
Validasi file JSON kamus sebelum deploy.

Usage:
    python scripts/validate_dict.py data/informal_map.json
    python scripts/validate_dict.py data/synonyms/bahan_protein.json
    python scripts/validate_dict.py --all
"""

import json
import sys
import os
import glob

REQUIRED_META_KEYS = ["description", "last_updated", "maintainer"]


def validate_file(path: str) -> tuple[bool, list[str]]:
    errors = []

    # 1. File exists?
    if not os.path.isfile(path):
        return False, [f"File tidak ditemukan: {path}"]

    # 2. Valid JSON?
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"JSON tidak valid: {e}"]
    except UnicodeDecodeError:
        return False, ["File harus berformat UTF-8"]

    # 3. Has _meta?
    meta = data.get("_meta", {})
    for key in REQUIRED_META_KEYS:
        if key not in meta:
            errors.append(f"_meta.{key} tidak ada")

    # 4. All values string?
    for k, v in data.items():
        if k.startswith("_"):
            continue  # skip meta & section markers
        if isinstance(v, list):
            for item in v:
                if not isinstance(item, str):
                    errors.append(f"'{k}': item list harus string, dapat {type(item)}")
        elif not isinstance(v, str):
            errors.append(f"'{k}': value harus string atau list, dapat {type(v)}")

    # 5. No duplicate values hint (warning only)
    warnings = []
    seen_keys = {}
    for k, v in data.items():
        if k.startswith("_"):
            continue
        v_str = str(v)
        if v_str in seen_keys and v_str:
            warnings.append(f"Duplikat value '{v_str}' di key '{k}' dan '{seen_keys[v_str]}'")
        seen_keys[v_str] = k

    return len(errors) == 0, errors + (["⚠ WARNING: " + w for w in warnings])


def main():
    if "--all" in sys.argv:
        files = (
            glob.glob("data/*.json") +
            glob.glob("data/synonyms/*.json")
        )
    elif len(sys.argv) > 1:
        files = [sys.argv[1]]
    else:
        print("Usage: python scripts/validate_dict.py <file.json>  OR  --all")
        sys.exit(1)

    all_ok = True
    for path in files:
        ok, messages = validate_file(path)
        status = "✓ OK" if ok else "✗ GAGAL"
        print(f"\n{status}  {path}")
        for msg in messages:
            print(f"   {'⚠' if 'WARNING' in msg else '→'} {msg}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("✅ Semua file valid. Aman untuk deploy.")
        sys.exit(0)
    else:
        print("❌ Ada error. Perbaiki dulu sebelum deploy.")
        sys.exit(1)


if __name__ == "__main__":
    main()