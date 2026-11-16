"""Edge-case matrix: what does the served model do with requests it should NOT simply accept?

    python -m s3_deployment.edge_cases [--url http://127.0.0.1:5001]

Method: every case carries a PREDICTION (accept / reject) written in this file BEFORE the server was
ever queried with it. Where the server disagrees, that is a finding: our mental model of the system
was wrong. Output: table + reports/edge_cases.md. This is a characterization, not a pass/fail gate:
exit code is 0 unless the server is unreachable.
"""
import argparse
import json
import sys

import numpy as np

import config
from s3_deployment.common import (
    MODEL_PORT, post, predictions, records_payload, sample_features, split_payload, url_for,
)

REPORTS = config.ROOT / "reports"


def build_cases(X) -> list:
    """(name, why_it_matters, predicted_outcome, body, content_type). predicted_outcome is
    'accept' (200 + valid probabilities) or 'reject' (HTTP error). Written before running."""
    row = X.iloc[[0]]
    rec = json.loads(row.to_json(orient="records", double_precision=15))[0]

    def recs(**changes):  # one-row records payload with fields changed / removed (value DROP removes)
        r = {**rec}
        for k, v in changes.items():
            if v == "DROP":
                r.pop(k, None)
            else:
                r[k] = v
        return {"dataframe_records": [r]}

    j = lambda d: json.dumps(d)
    return [
        ("valid single row",              "baseline", "accept", j(split_payload(row)), "application/json"),
        ("valid batch of 1000 rows",      "batching", "accept", j(split_payload(sample_features(1000, seed=1))), "application/json"),
        # FINDING (fixed in training code via strict_signature, but v1-v3 predate the fix): the old signature
        # marked workclass optional, MLflow let a request omit it, and the sklearn pipeline then crashed with
        # an opaque 400. For models retrained from now on this case is expected to be a clean schema rejection.
        ("optional column absent",        "workclass is required:false in the OLD signature", "accept", j(recs(workclass="DROP")), "application/json"),
        ("REQUIRED column absent (age)",  "schema enforcement", "reject", j(recs(age="DROP")), "application/json"),
        ("optional value null",           "missing workclass", "accept", j(recs(workclass=None)), "application/json"),
        ("required numeric null (age)",   "null in a required double", "accept", j(recs(age=None)), "application/json"),
        ("wrong type: age='forty'",       "type enforcement", "reject", j(recs(age="forty")), "application/json"),
        ("numeric string: age='39'",      "string that looks numeric", "reject", j(recs(age="39")), "application/json"),
        ("unseen category",               "workclass='Astronaut' never seen in training", "accept", j(recs(workclass="Astronaut")), "application/json"),
        ("extra column",                  "client sends a field the model does not use", "accept", j(recs(unused_field=1)), "application/json"),
        ("integer where double expected", "age sent as 39 not 39.0", "accept", j(recs(age=39)), "application/json"),
        ("OUT-OF-RANGE values",           "age=250, hours-per-week=-5: does anything check semantics?", "accept", j(recs(age=250.0, **{"hours-per-week": -5.0})), "application/json"),
        ("column order shuffled",         "dataframe_split with reversed columns", "accept",
         j(split_payload(row[list(row.columns)[::-1]])), "application/json"),
        ("zero rows",                     "empty batch", "accept", j({"dataframe_split": {"columns": list(row.columns), "data": []}}), "application/json"),
        ("'inputs' format, no names",     "positional list for a model with named columns", "reject", j({"inputs": [list(rec.values())]}), "application/json"),
        ("LEGACY Lab 1 payload",          "{'columns','instances'} from the old notebooks", "reject",
         j({"columns": list(row.columns), "instances": [list(rec.values())]}), "application/json"),
        ("malformed JSON",                "truncated body", "reject", '{"dataframe_split": {"columns": [', "application/json"),
        ("wrong Content-Type",            "text/plain", "reject", j(split_payload(row)), "text/plain"),
    ]


def classify(status: int, text: str) -> tuple[str, str]:
    """('accept'|'reject'|'accept-but-bad', detail)."""
    if status == 200:
        try:
            out = predictions(text)
        except Exception as e:
            return "accept-but-bad", f"200 but unparsable body: {e}"
        if out.size == 0:
            return "accept", "200, empty predictions []"
        if out.ndim != 2 or out.shape[1] != 2 or out.min() < 0 or out.max() > 1:
            return "accept-but-bad", f"200 but invalid output shape/range {out.shape}"
        return "accept", f"200, P(>50K)={out[0, 1]:.3f}" + (f" (+{len(out) - 1} more rows)" if len(out) > 1 else "")
    try:
        msg = json.loads(text).get("message") or json.loads(text).get("error_code") or text
    except Exception:
        msg = text
    return "reject", f"HTTP {status}: {str(msg).replace(chr(10), ' ')[:140]}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=url_for(MODEL_PORT))
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    X = sample_features(2000)
    rows, surprises = [], []
    for name, why, predicted, body, ctype in build_cases(X):
        status, text, secs = post(args.url, body, ctype)
        if status == 0:
            sys.exit(f"server unreachable at {args.url}: {text}")
        observed, detail = classify(status, text)
        match = observed == predicted
        rows.append((name, predicted, observed, "yes" if match else "NO <-- surprise", f"{secs * 1000:.0f}", detail))
        if not match:
            surprises.append((name, why, predicted, observed, detail))

    header = ("case", "predicted", "observed", "as predicted?", "ms", "detail")
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header) - 1)]
    lines = [" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r[:-1])) + " | " + r[-1] for r in [header] + rows]
    print("\n".join(lines))
    print(f"\n{len(rows) - len(surprises)}/{len(rows)} behaved as predicted; {len(surprises)} surprise(s):")
    for name, why, pred, obs, detail in surprises:
        print(f"  * {name}: predicted {pred}, got {obs}. {detail}\n      (why it matters: {why})")

    REPORTS.mkdir(exist_ok=True)
    md = ["# Edge cases\n", "| case | predicted | observed | as predicted? | detail |", "|---|---|---|---|---|"]
    md += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[5].replace('|', '/')} |" for r in rows]
    (REPORTS / "edge_cases.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
