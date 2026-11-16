"""Serve a registered model over REST with `mlflow models serve`.

    python -m s3_deployment.serve_local start [--uri models:/adult-income-classifier@champion] [--port 5001]
    python -m s3_deployment.serve_local status [--port 5001]
    python -m s3_deployment.serve_local stop   [--port 5001]

Needs the tracking server running (`python -m ops.start_server`) because models:/ URIs are resolved
through the registry. IMPORTANT: an alias is resolved ONCE, at startup. Promoting or rolling back
afterwards does not change what a running server answers with (alias_experiment.py proves it).
"""
import argparse
import sys

import mlflow

import config
from s2_registry.common import alias_version, init
from s3_deployment.common import MODEL_PORT, get, start_model_server, stop_model_server, url_for

DEFAULT_URI = f"models:/{config.REGISTERED_MODEL_NAME}@{config.CHAMPION_ALIAS}"


def resolved_version(uri: str) -> str | None:
    """Which registry version a models:/NAME@alias or models:/NAME/N URI points to RIGHT NOW."""
    client = init()
    if "@" in uri:
        name, alias = uri.removeprefix("models:/").split("@")
        mv = client.get_model_version_by_alias(name, alias)
        return mv.version
    parts = uri.removeprefix("models:/").split("/")
    return parts[1] if len(parts) == 2 and parts[1].isdigit() else None


def probe(port: int) -> None:
    base = url_for(port)
    for path in ("/ping", "/health", "/version"):
        status, body = get(f"{base}{path}")
        print(f"  GET {path:9s} -> {status} {body.strip()[:60]!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["start", "stop", "status"])
    p.add_argument("--uri", default=DEFAULT_URI)
    p.add_argument("--port", type=int, default=MODEL_PORT)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.cmd == "start":
        init()
        if get(f"{config.TRACKING_URI}/health", 2)[0] != 200:
            sys.exit("tracking server is not running: python -m ops.start_server")
        ver = resolved_version(args.uri)
        print(f"serving {args.uri}" + (f"  (resolves to v{ver} right now)" if ver else ""))
        info = start_model_server(args.uri, args.port, args.workers)
        print(f"healthy on {url_for(args.port)} after {info['seconds_to_healthy']:.1f}s (pid {info['pid']}, workers={args.workers})")
        probe(args.port)
    elif args.cmd == "status":
        status, _ = get(f"{url_for(args.port)}/health", 2)
        print(f"port {args.port}: {'UP' if status == 200 else 'DOWN'}")
        if status == 200:
            probe(args.port)
    else:
        print("stopped" if stop_model_server(args.port) else "no server recorded for that port",
              f"| port {args.port}", "still answering!" if get(f"{url_for(args.port)}/health", 2)[0] == 200 else "free")


if __name__ == "__main__":
    main()
