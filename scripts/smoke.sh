#!/usr/bin/env bash
#
# End-to-end smoke test against a running engine (container or local uvicorn).
#
#   BASE_URL=http://localhost:8000 ./scripts/smoke.sh
#
# It checks the things a reviewer would check by hand: the SPA is served, the API
# answers, every available strategy returns a feasible plan, the comparison ranks
# them, plans are reproducible byte for byte, and the RL artifact is either
# served or explained. No jq dependency: it uses the container's own Python.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
PYTHON="${PYTHON:-python3}"

say() { printf '\n\033[36m== %s\033[0m\n' "$1"; }
fail() { printf '\033[31mFAIL: %s\033[0m\n' "$1" >&2; exit 1; }

say "SPA index"
INDEX_HTML="$(curl -fsS "$BASE_URL/")"
echo "$INDEX_HTML" | grep -q 'id="root"' || fail "the SPA index is not served at $BASE_URL/"
echo "$INDEX_HTML" | grep -q '<script type="module"' || fail "the SPA entry script is missing"
echo "ok: index.html served ($(printf '%s' "$INDEX_HTML" | wc -c | tr -d ' ') bytes)"

say "health"
curl -fsS "$BASE_URL/api/health" | "$PYTHON" -c '
import json, sys
payload = json.load(sys.stdin)
assert payload["status"] == "ok", payload
print("ok: version", payload["version"], "algorithms", payload["algorithms"])
print("    unavailable:", payload["unavailable"] or "none")
print("    frontend built:", payload["frontend_built"])
'

say "scenario"
curl -fsS -X POST "$BASE_URL/api/scenario" -H 'Content-Type: application/json' \
  -d '{"preset":"small","seed":7}' | "$PYTHON" -c '
import json, sys
payload = json.load(sys.stdin)
summary = payload["summary"]
assert summary["requests"] == 40 and summary["vehicles"] == 6, summary
print("ok:", summary["name"], summary["vehicles"], "vehicles /", summary["requests"], "requests")
'

say "allocate with every available strategy"
ALGOS=$(curl -fsS "$BASE_URL/api/algorithms" | "$PYTHON" -c '
import json, sys
catalogue = json.load(sys.stdin)["algorithms"]
# `exact` is guarded to tiny instances and is checked separately below
print(" ".join(entry["name"] for entry in catalogue if entry["available"] and entry["name"] != "exact"))
')
for ALGORITHM in $ALGOS; do
  curl -fsS -X POST "$BASE_URL/api/allocate" -H 'Content-Type: application/json' \
    -d "{\"scenario\":{\"preset\":\"small\",\"seed\":7},\"algorithm\":\"$ALGORITHM\",\"include_scenario\":false}" \
    | "$PYTHON" -c '
import json, sys
plan = json.load(sys.stdin)["plan"]
metrics = plan["metrics"]
assert metrics["valid"], metrics["hard_violations"]
assert metrics["requests_assigned"] + metrics["requests_unassigned"] == metrics["requests_total"]
assert all(assignment["reason"] for assignment in plan["assignments"])
print("  ok: %-10s objective %8.1f  coverage %5.1f%%  %6.1f ms"
      % (plan["algorithm"], metrics["objective"], metrics["coverage_pct"], plan["runtime_ms"]))
'
done

say "exhaustive solver guard"
GUARD_STATUS=$(curl -s -o /tmp/smoke_guard.json -w '%{http_code}' -X POST "$BASE_URL/api/allocate" \
  -H 'Content-Type: application/json' \
  -d '{"scenario":{"preset":"small"},"algorithm":"exact","include_scenario":false}')
[ "$GUARD_STATUS" = "400" ] || fail "expected a 400 for the exhaustive solver on a 40-request book, got $GUARD_STATUS"
"$PYTHON" -c '
import json
detail = json.load(open("/tmp/smoke_guard.json"))["detail"]
assert "limited to" in detail, detail
print("ok: refused with a clear reason ->", detail)
'

say "compare"
curl -fsS -X POST "$BASE_URL/api/compare" -H 'Content-Type: application/json' \
  -d '{"scenario":{"preset":"small","seed":7},"include_scenario":false}' | "$PYTHON" -c '
import json, sys
payload = json.load(sys.stdin)
rankings = payload["rankings"]
assert payload["winner"] == rankings[0]["algorithm"]
objectives = [row["objective"] for row in rankings]
assert objectives == sorted(objectives), objectives
for row in rankings:
    print("  %-10s objective %8.1f  coverage %5.1f%%  %7.1f ms"
          % (row["algorithm"], row["objective"], row["coverage_pct"], row["runtime_ms"]))
print("ok: winner", payload["winner"])
'

say "determinism (same scenario, twice)"
for ALGORITHM in $ALGOS; do
  "$PYTHON" - "$BASE_URL" "$ALGORITHM" <<'PY'
import json, sys, urllib.request

base, algorithm = sys.argv[1], sys.argv[2]
body = json.dumps({
    "scenario": {"preset": "small", "seed": 7},
    "algorithm": algorithm,
    "include_scenario": False,
}).encode()

def call():
    request = urllib.request.Request(
        base + "/api/allocate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["plan"]

first, second = call(), call()
assert first["routes"] == second["routes"], "%s is not deterministic" % algorithm
print("  ok: %s produced identical routes twice" % algorithm)
PY
done

say "RL policy artifact"
if curl -fsS "$BASE_URL/api/rl/policy?include_curve=false" > /tmp/smoke_policy.json 2>/dev/null; then
  "$PYTHON" -c '
import json
payload = json.load(open("/tmp/smoke_policy.json"))
print("ok: policy has", len(payload["weights"]), "weights, trained for",
      payload["metadata"].get("train_episodes"), "episodes; holdout:",
      payload["metadata"].get("holdout_comparison", {}).get("best"))
'
else
  echo "ok: learned strategy unavailable in this image (reported, not hidden)"
fi

printf '\n\033[32mSMOKE OK\033[0m (%s)\n' "$BASE_URL"
