"""Translate a wetlab release's own training_split.json (schema
``wetlab_explicit_split_v1`` or ``wetlab_capture_group_split_v1``, both with
top-level ``train``/``validation``/``test`` uuid lists) into the manifest
shape ``build_wetlab_canonical_dataset.py``'s ``split_uuids()`` expects: a
dict with ``train``, ``val``, ``block_holdout_v1.holdout_episodes`` and
``block_of_episode``.

Neither known release ships a held-out test split (their own ``test`` lists
are empty), so ``block_holdout_v1.holdout_episodes`` is written empty here
and every uuid maps to itself in ``block_of_episode`` -- that field is only
consumed by this project's own disjointness/grouping bookkeeping (see
``merge()`` in ``build_wetlab_canonical_dataset.py``), not something the
publisher's fixed split needs to satisfy beyond being present.

Usage:
  python scripts/translate_smoketestv2_split.py \
    /DATA2/qianqian/n0vtla_robot_audit/smoke_test_v2_docs/training_split.json \
    scripts/wetlab_split_smoketestv2_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KNOWN_SCHEMAS = {"wetlab_explicit_split_v1", "wetlab_capture_group_split_v1"}


def translate(source: dict) -> dict:
    if source.get("schema") not in KNOWN_SCHEMAS:
        raise ValueError(f"Unexpected training_split.json schema: {source.get('schema')!r} "
                          f"(known: {sorted(KNOWN_SCHEMAS)}) -- verify train/validation/test "
                          f"still mean the same thing in this release before adding it here")
    train = list(source["train"])
    val = list(source["validation"])
    if source["test"]:
        raise ValueError("This release's own test split is non-empty -- this translator "
                          "assumes no held-out test split; update it before trusting the output")
    if not set(train).isdisjoint(val):
        raise ValueError("train/validation overlap in the source split -- refusing to translate")
    all_uuids = train + val
    if len(set(all_uuids)) != len(all_uuids):
        raise ValueError("duplicate uuids across train/validation in the source split")
    return {
        "train": train,
        "val": val,
        "block_holdout_v1": {"holdout_episodes": []},
        "block_of_episode": {u: u for u in all_uuids},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("training_split_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source = json.loads(args.training_split_json.read_text())
    manifest = translate(source)
    args.output.write_text(json.dumps(manifest, indent=2))
    print(f"train={len(manifest['train'])} val={len(manifest['val'])} -> {args.output}")


if __name__ == "__main__":
    main()
