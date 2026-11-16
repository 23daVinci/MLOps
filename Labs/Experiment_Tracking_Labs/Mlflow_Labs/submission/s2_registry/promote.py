"""Registry step 2: gated promotion, challenger workflow, deprecation and rollback via aliases.

    python -m s2_registry.promote <version>            # try to make <version> the champion
    python -m s2_registry.promote --challenge <ver>    # mark as 'challenger' (passes gates 1-2, no contest)
    python -m s2_registry.promote --check <ver>        # run gates 1-2 only, change no alias (audit / backfill)
    python -m s2_registry.promote --deprecate <ver> --reason "..."
    python -m s2_registry.promote --rollback           # champion <-> previous-champion

A version becomes champion only if it passes three gates, in order (cheapest evidence first):
  1. CONTRACT     loaded through the registry (pyfunc) it returns valid probabilities and
                  reproduces the AUC that was logged during tracking.
  2. BUDGETS      single-row p95 latency and model size are within the operational budget
                  (a model that is 0.002 AUC better but 10x slower is not a free upgrade).
  3. PERFORMANCE  it beats the current champion's validation AUC by PROMOTION_MARGIN
                  (the first-ever champion has nothing to beat and skips this gate).
The test split is NOT used here; it stays held out for the evaluation stage.
"""
import argparse
import sys
import time
from datetime import datetime, timezone

import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score

import config
import data
from s2_registry.common import (
    alias_version, init, model_size_bytes, record_failure, run_metric, set_status,
)

TOLERANCE = 1e-3
NAME = config.REGISTERED_MODEL_NAME


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def refuse_if_interrupted(client) -> None:
    """A rollback writes a 'pending_rollback' tag before touching aliases and removes it after.
    If it is still there, a previous rollback crashed between the two alias writes: the aliases
    may be inconsistent, so do NOT stack another change on top until a human has looked."""
    tags = client.get_registered_model(NAME).tags
    if "pending_rollback" in tags:
        sys.exit(
            f"REFUSING: an earlier rollback ({tags['pending_rollback']}) did not finish.\n"
            f"  champion={getattr(alias_version(client, config.CHAMPION_ALIAS), 'version', None)}, "
            f"previous-champion={getattr(alias_version(client, config.PREVIOUS_ALIAS), 'version', None)}\n"
            "  Fix the aliases by hand, then delete the registered-model tag 'pending_rollback'."
        )


# ------------------------------------------------------------------ gates ---
def check_contract(client, mv, model, X_val, y_val) -> tuple[bool, str]:
    out = np.asarray(model.predict(X_val))
    if out.ndim != 2 or out.shape != (len(X_val), 2):
        return False, f"output shape {out.shape}, expected ({len(X_val)}, 2)"
    if out.min() < 0 or out.max() > 1 or not np.allclose(out.sum(axis=1), 1.0, atol=1e-6):
        return False, "outputs are not valid probabilities"
    auc = roc_auc_score(y_val, out[:, 1])
    logged = run_metric(client, mv)
    if abs(auc - logged) > TOLERANCE:
        return False, f"AUC after loading {auc:.4f} != logged {logged:.4f}"
    return True, f"valid probabilities, AUC {auc:.4f} reproduces logged {logged:.4f}"


def measure_latency_p95_ms(model, X, n: int = 150, warmup: int = 10) -> float:
    """p95 of single-row predict() calls, in-process (excludes network; it is a regression budget,
    not a promise about end-to-end latency, which Stage 3 measures over HTTP)."""
    rows = [X.iloc[[i]] for i in range(n + warmup)]
    for r in rows[:warmup]:  # first calls pay one-time costs (imports, allocations)
        model.predict(r)
    times = []
    for r in rows[warmup:]:
        t0 = time.perf_counter()
        model.predict(r)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 95))


def check_budgets(client, mv, model, X_val) -> tuple[bool, str]:
    p95 = measure_latency_p95_ms(model, X_val)
    size_mb = model_size_bytes(NAME, mv.version) / 1e6
    # measurements are facts, not verdicts: record them even if a budget is blown
    client.set_model_version_tag(NAME, mv.version, "latency_p95_ms", f"{p95:.2f}")
    client.set_model_version_tag(NAME, mv.version, "model_size_mb", f"{size_mb:.3f}")
    problems = []
    if p95 > config.LATENCY_P95_MS_BUDGET:
        problems.append(f"p95 latency {p95:.1f} ms > budget {config.LATENCY_P95_MS_BUDGET:g} ms")
    if size_mb > config.MODEL_SIZE_MB_BUDGET:
        problems.append(f"size {size_mb:.1f} MB > budget {config.MODEL_SIZE_MB_BUDGET:g} MB")
    msg = f"p95 {p95:.1f} ms (budget {config.LATENCY_P95_MS_BUDGET:g}), size {size_mb:.2f} MB (budget {config.MODEL_SIZE_MB_BUDGET:g})"
    return (not problems), ("; ".join(problems) if problems else msg)


def run_gates_1_and_2(client, mv):
    """Shared by promote and challenge. Exits (after recording why) on failure."""
    _, _, _, val, _ = data.get_splits()
    X_val, y_val = data.xy(val)
    model = mlflow.pyfunc.load_model(f"models:/{NAME}/{mv.version}")  # pinned, by version

    ok, msg = check_contract(client, mv, model, X_val, y_val)
    print(f"gate 1 (contract):    {'PASS' if ok else 'FAIL'} - {msg}")
    if not ok:
        record_failure(client, mv, "failed", f"contract: {msg}")
        sys.exit(1)

    ok, msg = check_budgets(client, mv, model, X_val)
    print(f"gate 2 (budgets):     {'PASS' if ok else 'FAIL'} - {msg}")
    if not ok:
        record_failure(client, mv, "failed", f"budgets: {msg}")
        sys.exit(1)


def get_version(client, version: str):
    mv = client.get_model_version(NAME, version)
    if mv.tags.get("deprecated") == "true":
        sys.exit(f"version {version} is deprecated ({mv.tags.get('deprecated_reason', 'no reason recorded')}); refusing.")
    return mv


# --------------------------------------------------------------- commands ---
def promote(version: str) -> None:
    client = init()
    refuse_if_interrupted(client)
    mv = get_version(client, version)
    champion = alias_version(client, config.CHAMPION_ALIAS)
    if champion is not None and champion.version == mv.version:
        sys.exit(f"version {version} is already champion")

    run_gates_1_and_2(client, mv)

    if champion is not None:
        cand, base = run_metric(client, mv), run_metric(client, champion)
        gain = cand - base
        ok = gain >= config.PROMOTION_MARGIN
        note = (f"candidate {cand:.4f} vs champion v{champion.version} {base:.4f} "
                f"(gain {gain:+.4f}, need >= {config.PROMOTION_MARGIN})")
        print(f"gate 3 (performance): {'PASS' if ok else 'FAIL'} - {note}")
        if not ok:
            record_failure(client, mv, "rejected", f"performance: {note}")
            sys.exit(1)
        client.set_registered_model_alias(NAME, config.PREVIOUS_ALIAS, champion.version)
    else:
        note = "bootstrap champion: contract and budgets OK, no incumbent to beat"
        print("gate 3 (performance): SKIP - no current champion (bootstrap)")

    client.set_registered_model_alias(NAME, config.CHAMPION_ALIAS, mv.version)
    set_status(client, mv, "passed", note)
    client.set_model_version_tag(NAME, mv.version, "promoted_at", now())
    # promoting the challenger ends its candidacy
    ch = alias_version(client, config.CHALLENGER_ALIAS)
    if ch is not None and ch.version == mv.version:
        client.delete_registered_model_alias(NAME, config.CHALLENGER_ALIAS)
    print(f"-> '{config.CHAMPION_ALIAS}' now points to version {mv.version}"
          + (f" (previous: v{champion.version})" if champion else ""))


def check(version: str) -> None:
    """Read-only audit: run gates 1-2 and record the measurements (latency, size) as tags.
    Moves no alias. Use it to backfill versions that were promoted before a gate existed."""
    client = init()
    mv = client.get_model_version(NAME, version)  # deprecated versions can still be audited
    run_gates_1_and_2(client, mv)
    print(f"v{version}: checks passed; measurements recorded; no alias changed")


def challenge(version: str) -> None:
    """Mark a version as the challenger: it must be a sound model (gates 1-2) but does not have to
    beat the champion yet. Stage 3's shadow comparison scores live-like traffic with both."""
    client = init()
    refuse_if_interrupted(client)
    mv = get_version(client, version)
    for alias in (config.CHAMPION_ALIAS, config.PREVIOUS_ALIAS):
        held = alias_version(client, alias)
        if held is not None and held.version == mv.version:
            sys.exit(f"version {version} is already '{alias}'; it cannot also be the challenger")
    run_gates_1_and_2(client, mv)
    client.set_registered_model_alias(NAME, config.CHALLENGER_ALIAS, mv.version)
    client.set_model_version_tag(NAME, mv.version, "challenger_since", now())
    print(f"-> '{config.CHALLENGER_ALIAS}' now points to version {mv.version}")


def deprecate(version: str, reason: str) -> None:
    """Retire a version without deleting it: versions that ever served traffic must stay traceable."""
    client = init()
    refuse_if_interrupted(client)
    mv = client.get_model_version(NAME, version)
    for alias in (config.CHAMPION_ALIAS, config.PREVIOUS_ALIAS):
        held = alias_version(client, alias)
        if held is not None and held.version == mv.version:
            sys.exit(f"REFUSING: version {version} holds '{alias}'. Promote or roll back first, then deprecate.")
    ch = alias_version(client, config.CHALLENGER_ALIAS)
    if ch is not None and ch.version == mv.version:
        client.delete_registered_model_alias(NAME, config.CHALLENGER_ALIAS)
        print(f"removed '{config.CHALLENGER_ALIAS}' alias from v{version}")
    client.set_model_version_tag(NAME, mv.version, "deprecated", "true")
    client.set_model_version_tag(NAME, mv.version, "deprecated_reason", reason[:250])
    client.set_model_version_tag(NAME, mv.version, "deprecated_at", now())
    print(f"v{version} deprecated: {reason}  (still in the registry; never deleted)")


def rollback() -> None:
    client = init()
    refuse_if_interrupted(client)
    current = alias_version(client, config.CHAMPION_ALIAS)
    previous = alias_version(client, config.PREVIOUS_ALIAS)
    if current is None or previous is None:
        sys.exit("Nothing to roll back to (need both a champion and a previous-champion).")
    if previous.tags.get("deprecated") == "true":
        sys.exit(f"REFUSING: previous-champion v{previous.version} is deprecated; pick a version with `promote` instead.")
    # Two alias writes can't be atomic, so order matters. Champion goes FIRST: if we crash between
    # the writes, `champion` already points at the good version (what the operator wanted) and only
    # the undo pointer is lost. The reverse order would leave users on the bad version AND lose the
    # pointer to the good one. A write-ahead tag records the intent so a crash is detectable.
    client.set_registered_model_tag(NAME, "pending_rollback", f"v{current.version}->v{previous.version}")
    client.set_registered_model_alias(NAME, config.CHAMPION_ALIAS, previous.version)
    client.set_registered_model_alias(NAME, config.PREVIOUS_ALIAS, current.version)  # allows undoing the rollback
    client.delete_registered_model_tag(NAME, "pending_rollback")
    # A rollback is an event, not a validation result: record it separately and leave
    # validation_status (did it pass its gates?) untouched.
    client.set_model_version_tag(NAME, current.version, "last_rollback_at", now())
    print(f"rolled back: '{config.CHAMPION_ALIAS}' v{current.version} -> v{previous.version} "
          f"('{config.PREVIOUS_ALIAS}' now v{current.version})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("version", nargs="?", help="model version to promote to champion")
    p.add_argument("--rollback", action="store_true")
    p.add_argument("--challenge", metavar="VERSION")
    p.add_argument("--check", metavar="VERSION")
    p.add_argument("--deprecate", metavar="VERSION")
    p.add_argument("--reason", help="required with --deprecate")
    args = p.parse_args()

    chosen = [bool(args.version), args.rollback, bool(args.challenge), bool(args.deprecate), bool(args.check)]
    if sum(chosen) != 1:
        p.error("give exactly one of: <version>, --rollback, --challenge V, --check V, --deprecate V")
    if args.deprecate and not args.reason:
        p.error("--deprecate needs --reason (an unexplained deprecation is useless in six months)")

    if args.rollback:
        rollback()
    elif args.challenge:
        challenge(args.challenge)
    elif args.check:
        check(args.check)
    elif args.deprecate:
        deprecate(args.deprecate, args.reason)
    else:
        promote(args.version)


if __name__ == "__main__":
    main()
