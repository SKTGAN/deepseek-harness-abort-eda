"""Recover a loss_decoder program from its saved capsule and training matrix."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


def checked_source(capsule, matrix_id, candidate_id):
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
               for value in (matrix_id, candidate_id)):
        raise ValueError("matrix and candidate IDs must be full lowercase SHA-256 values")
    files = capsule["candidate_files"]
    if set(files) != {"loss_decoder_program.json"}:
        raise ValueError("expected exactly one loss_decoder candidate program")
    source = files["loss_decoder_program.json"]["source"]
    if not isinstance(source, str):
        raise ValueError("candidate source must be text")
    identity = {"task": "tp_gnn_loss_decoder_timing", "matrix": matrix_id, "source": source}
    actual = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != candidate_id:
        raise ValueError(f"candidate identity mismatch: calculated {actual}")
    compile(source, "program.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path, help="saved candidates/<id>/candidate.json")
    parser.add_argument("matrix", type=Path, help="matrix.json from the matching training evaluation")
    parser.add_argument("--id", required=True, help="full candidate ID from that evaluation receipt")
    parser.add_argument("--out", type=Path, required=True, help="new output file, typically program.py")
    args = parser.parse_args()
    try:
        capsule = json.loads(args.capsule.read_text(encoding="utf-8"))
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        source = checked_source(capsule, matrix["matrix_sha256"], args.id)
        with args.out.open("xb") as output:
            output.write(source.encode("utf-8"))
    except (OSError, KeyError, TypeError, ValueError, SyntaxError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Verified and exported {args.id} to {args.out}")


if __name__ == "__main__":
    main()
