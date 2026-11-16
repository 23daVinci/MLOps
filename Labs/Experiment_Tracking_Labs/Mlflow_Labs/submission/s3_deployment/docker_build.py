"""Build, run and VERIFY a container image of one registered model version.

    python -m s3_deployment.docker_build [--version N] [--port 5003] [--keep]

What it does, in order:
  1. downloads the registered version (pinned by number, never an alias) into build/v<N>/model
  2. docker build -> image `adult-income:v<N>`   (image tag = model version = what is actually inside)
  3. docker run, measures COLD START (container start -> /health OK)
  4. runs the SAME smoke / parity / load tests used on the local server, against the container
  5. checks it runs as non-root, prints image size, removes the container (unless --keep)
Writes reports/docker_report.md. Requires Docker Desktop running; the tracking server for the download.
"""
import argparse
import shutil
import subprocess
import sys
import time

import mlflow

import config
from s2_registry.common import alias_version, init
from s3_deployment import load_test, parity_test, smoke_test
from s3_deployment.common import get, sample_features, url_for, wait_healthy

NAME = config.REGISTERED_MODEL_NAME
BUILD = config.ROOT / "build"
REPORTS = config.ROOT / "reports"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def sh(cmd: list, log=None, check=True) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, stdout=log or subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=config.ROOT)
    if check and r.returncode != 0:
        tail = "" if log else (r.stdout or "")[-1500:]
        raise RuntimeError(f"{' '.join(cmd[:3])}... failed (exit {r.returncode})\n{tail}")
    return r


def prepare_context(version: str):
    ctx = BUILD / f"v{version}"
    if ctx.exists():
        shutil.rmtree(ctx)
    ctx.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{NAME}/{version}", dst_path=str(ctx / "model"))
    shutil.copy(config.ROOT / "s3_deployment" / "Dockerfile", ctx / "Dockerfile")
    reqs = (ctx / "model" / "requirements.txt").read_text()
    return ctx, reqs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", help="registered version to containerize (default: current champion)")
    p.add_argument("--port", type=int, default=5003)
    p.add_argument("--keep", action="store_true", help="leave the container running afterwards")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    client = init()
    version = args.version or alias_version(client, config.CHAMPION_ALIAS).version
    image, cname = f"adult-income:v{version}", f"adult-income-v{version}"
    report = {}

    if sh(["docker", "info"], check=False).returncode != 0:
        sys.exit("Docker daemon is not reachable: start Docker Desktop")

    # ---- 1-2. context + build ---------------------------------------------------------------
    ctx, reqs = prepare_context(version)
    build_args = ["--build-arg", f"PIP_EXTRA_INDEX={TORCH_CPU_INDEX}"] if "torch" in reqs else []
    print(f"building {image} from registered v{version}; requirements: {' '.join(reqs.split())}")
    (config.ROOT / "logs").mkdir(exist_ok=True)
    log_path = config.ROOT / "logs" / f"docker_build_v{version}.log"
    t0 = time.time()
    with open(log_path, "w") as log:
        r = sh(["docker", "build", "-t", image, *build_args, str(ctx)], log=log, check=False)
    report["build_seconds"] = time.time() - t0
    if r.returncode != 0:
        print("".join(open(log_path).readlines()[-25:]))
        sys.exit(f"docker build FAILED (log: {log_path})")
    report["image_mb"] = int(sh(["docker", "image", "inspect", image, "--format", "{{.Size}}"]).stdout.strip()) / 1e6
    # The two size commands count DIFFERENTLY (compressed content vs unpacked disk usage); report both
    # instead of picking the flattering one.
    report["image_disk"] = sh(["docker", "images", image, "--format", "{{.Size}}"]).stdout.strip()
    print(f"built in {report['build_seconds']:.0f}s; size: `docker image inspect` {report['image_mb']:.0f} MB, "
          f"`docker images` {report['image_disk']} (compressed vs unpacked)")

    # ---- 3. run + cold start -----------------------------------------------------------------
    sh(["docker", "rm", "-f", cname], check=False)
    url = url_for(args.port)
    try:
        t0 = time.time()
        sh(["docker", "run", "-d", "--name", cname, "-p", f"{args.port}:8080", image])
        try:
            wait_healthy(url, timeout_s=180)
        except TimeoutError:
            print(sh(["docker", "logs", "--tail", "40", cname], check=False).stdout)
            raise
        report["cold_start_seconds"] = time.time() - t0
        print(f"container healthy {report['cold_start_seconds']:.1f}s after `docker run` (local server took ~25s)")

        report["user"] = sh(["docker", "exec", cname, "id", "-u"]).stdout.strip()
        # docker's own HEALTHCHECK should agree with ours (poll briefly: it runs every 15s)
        health = "unknown"
        for _ in range(20):
            health = sh(["docker", "inspect", "--format", "{{.State.Health.Status}}", cname]).stdout.strip()
            if health == "healthy":
                break
            time.sleep(2)
        report["docker_health"] = health
        print(f"runs as uid {report['user']} ({'non-root OK' if report['user'] != '0' else 'ROOT: bad'}); docker HEALTHCHECK = {health}")

        # ---- 4. the same tests as on the local server -----------------------------------------
        s = smoke_test.run(url)
        report["smoke"] = s["ok"]
        print(f"\nsmoke:  {'PASS' if s['ok'] else 'FAIL ' + str(s['problems'])}")
        pr = parity_test.run(url, version, 1000)
        report["parity"] = pr["ok"]
        for fmt, res in pr["results"].items():
            print(f"parity: {fmt:18s} {'PASS' if res['ok'] else 'FAIL'}  {res['detail']}")

        pool = sample_features(3000, seed=3)
        rows = []
        for batch, n in [(1, 100), (100, 50), (1000, 20)]:
            m = load_test.measure(url, load_test.make_bodies(pool, batch, n), 1)
            rows.append((batch, m))
            print(f"load:   batch={batch:<5d} p50={m['p50_ms']:6.1f}  p95={m['p95_ms']:6.1f} ms  {m['req_per_s'] * batch:9.0f} rows/s  errors={m['errors']}")
        report["load"] = rows

        local = f"{url_for(5001)}"
        if get(f"{local}/health", 2)[0] == 200:
            print("\nsame requests against the LOCAL server (:5001) for comparison:")
            for batch, n in [(1, 100), (1000, 20)]:
                m = load_test.measure(local, load_test.make_bodies(pool, batch, n), 1)
                print(f"        batch={batch:<5d} p50={m['p50_ms']:6.1f}  p95={m['p95_ms']:6.1f} ms  {m['req_per_s'] * batch:9.0f} rows/s")
    finally:
        if not args.keep:
            sh(["docker", "rm", "-f", cname], check=False)
            print(f"\nremoved container {cname}")

    ok = report.get("smoke") and report.get("parity") and report.get("user") != "0"
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "docker_report.md").write_text(
        f"# Docker deployment of registered v{version}\n\n- image: {image} (`docker image inspect`: {report['image_mb']:.0f} MB, "
        f"`docker images`: {report['image_disk']}; compressed vs unpacked)\n"
        f"- build: {report['build_seconds']:.0f}s\n- cold start: {report['cold_start_seconds']:.1f}s\n"
        f"- user: uid {report['user']}\n- docker healthcheck: {report['docker_health']}\n"
        f"- smoke: {report['smoke']}, parity vs in-process: {report['parity']}\n", encoding="utf-8")
    print(f"\nRESULT: {'container verified' if ok else 'PROBLEMS FOUND'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
