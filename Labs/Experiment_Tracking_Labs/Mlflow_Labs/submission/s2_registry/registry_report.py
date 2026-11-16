"""Lineage report: for every registered version, where it came from and whether it can be trusted.

    python -m s2_registry.registry_report

Answers the questions an on-call engineer or auditor asks: which version is serving, which run and
data produced it, was the code committed, did it pass its gates, how big and how fast is it?
Writes reports/registry_report.md (gitignored).
"""
import mlflow
import pandas as pd

import config
from s2_registry.common import aliases_by_version, init, source_run


def version_rows(client, name: str) -> pd.DataFrame:
    aliases = aliases_by_version(client, name)
    rows = []
    for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)):
        run = source_run(client, mv)
        # Source-run facts are the system of record; version tags are copies made at registration time.
        rows.append({
            "ver": f"v{mv.version}",
            "aliases": ",".join(sorted(aliases.get(mv.version, []))) or "-",
            "family": mv.tags.get("model_family", "-"),
            "status": mv.tags.get("validation_status", "-") + (" (deprecated)" if mv.tags.get("deprecated") == "true" else ""),
            "val_auc": f"{run.data.metrics['val_auc']:.4f}" if run and "val_auc" in run.data.metrics else mv.tags.get("val_auc", "-"),
            "p95_ms": mv.tags.get("latency_p95_ms", "-"),
            "size_mb": mv.tags.get("model_size_mb", "-"),
            "data": (run.data.tags.get("data_version") if run else None) or mv.tags.get("data_version", "-"),
            "commit": ((run.data.tags.get("mlflow.source.git.commit") if run else None) or mv.tags.get("git_commit", "-"))[:7],
            "git_dirty": (run.data.tags.get("git_dirty") if run else None) or mv.tags.get("git_dirty", "unknown"),
            "run": mv.run_id[:8] if mv.run_id else "-",
        })
    return pd.DataFrame(rows)


def findings(client, name: str, df: pd.DataFrame) -> list:
    out = []
    reg = client.get_registered_model(name)
    if config.CHAMPION_ALIAS not in reg.aliases:
        out.append("no champion: nothing is promoted yet")
    for _, r in df.iterrows():
        if r["git_dirty"] == "true":
            out.append(f"{r['ver']}: trained from an UNCOMMITTED working tree: commit {r['commit']} cannot reproduce it")
        elif r["git_dirty"] == "unknown":
            out.append(f"{r['ver']}: git state at training time unknown (run predates the git_dirty tag)")
        if r["status"].startswith("pending"):
            out.append(f"{r['ver']}: never gated (status pending)")
    if "pending_rollback" in reg.tags:
        out.append(f"INTERRUPTED ROLLBACK ({reg.tags['pending_rollback']}): aliases may be inconsistent")
    return out


def main() -> None:
    client = init()
    sections = []
    for name in (config.REGISTERED_MODEL_NAME, config.PROD_REGISTERED_MODEL_NAME):
        try:
            client.get_registered_model(name)
        except mlflow.exceptions.MlflowException:
            continue
        df = version_rows(client, name)
        text = f"## {name}\n\n{df.to_string(index=False)}\n"
        notes = findings(client, name, df)
        if notes:
            text += "\nFindings:\n" + "\n".join(f"  - {n}" for n in notes) + "\n"
        sections.append(text)
        print(text)
    out = config.ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "registry_report.md").write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {out / 'registry_report.md'}")


if __name__ == "__main__":
    main()
