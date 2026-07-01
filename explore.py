"""
Data exploration script for the Redrob candidate dataset.
Streams candidates.jsonl line by line — never loads the full 465MB into memory.
Run: python redrob-ranker/explore.py
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "candidates.jsonl"

# ── helpers ──────────────────────────────────────────────────────────────────

INDIA_CITIES = {"pune", "noida", "hyderabad", "mumbai", "delhi"}
TODAY = date(2026, 7, 1)   # fixed for reproducibility


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def days_since(d):
    if d is None:
        return None
    return (TODAY - d).days


# ── accumulators ─────────────────────────────────────────────────────────────

title_counter = Counter()

yoe_buckets = Counter()          # bucket label -> count
YOE_EDGES = [0, 2, 5, 8, 12, 17, 25, 50]

def yoe_label(y):
    for lo, hi in zip(YOE_EDGES, YOE_EDGES[1:]):
        if lo <= y < hi:
            return f"{lo}-{hi}"
    return "50+"

india_count = 0
non_india_count = 0
city_counter = Counter()

rrr_hist = Counter()             # recruiter_response_rate buckets (0.1 width)
last_active_hist = Counter()     # days-since-active buckets
open_to_work_counter = Counter()
icr_hist = Counter()             # interview_completion_rate buckets

impossible_flags = []            # up to 5 examples

total = 0

# ── single streaming pass ────────────────────────────────────────────────────

with open(DATA_PATH, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue

        total += 1

        profile   = c.get("profile", {})
        signals   = c.get("redrob_signals", {})
        skills    = c.get("skills", [])
        career    = c.get("career_history", [])

        # 1. current_title
        title = profile.get("current_title", "").strip()
        if title:
            title_counter[title] += 1

        # 2. years_of_experience
        yoe = profile.get("years_of_experience")
        if yoe is not None:
            yoe_buckets[yoe_label(float(yoe))] += 1

        # 3. location
        country  = (profile.get("country") or "").strip().lower()
        location = (profile.get("location") or "").strip().lower()

        if country == "india":
            india_count += 1
        else:
            non_india_count += 1

        for city in INDIA_CITIES:
            if city in location:
                city_counter[city] += 1

        # 4. behavioral signals
        rrr = signals.get("recruiter_response_rate")
        if rrr is not None:
            bucket = min(int(float(rrr) * 10), 9)   # 0-9
            rrr_hist[bucket] += 1

        last_active = parse_date(signals.get("last_active_date"))
        dsa = days_since(last_active)
        if dsa is not None:
            if dsa < 30:
                last_active_hist["<30d"] += 1
            elif dsa < 90:
                last_active_hist["30-90d"] += 1
            elif dsa < 180:
                last_active_hist["90-180d"] += 1
            elif dsa < 365:
                last_active_hist["180-365d"] += 1
            else:
                last_active_hist[">365d"] += 1

        otw = signals.get("open_to_work_flag")
        open_to_work_counter[bool(otw)] += 1

        icr = signals.get("interview_completion_rate")
        if icr is not None:
            bucket = min(int(float(icr) * 5), 4)    # quintiles 0-4
            icr_hist[bucket] += 1

        # 5. impossible profile flags (collect up to 5 examples)
        if len(impossible_flags) < 5:
            cid = c.get("candidate_id", "?")

            for sk in skills:
                if (sk.get("proficiency") == "expert"
                        and sk.get("duration_months", 1) == 0):
                    impossible_flags.append({
                        "id":     cid,
                        "reason": f"skill '{sk['name']}' marked 'expert' but duration_months=0",
                        "detail": sk
                    })
                    break

            if len(impossible_flags) < 5:
                for job in career:
                    dur = job.get("duration_months", 0)
                    company = job.get("company", "")
                    # Sentinel companies: Dunder Mifflin (fictional paper co, ~1949+)
                    # Flag any role whose duration > 600 months (50 years) as absurd
                    if dur > 600:
                        impossible_flags.append({
                            "id":     cid,
                            "reason": f"role at '{company}' has duration_months={dur} (>50 yrs)",
                            "detail": job
                        })
                        break

                    # Flag start_date after end_date
                    start = parse_date(job.get("start_date"))
                    end   = parse_date(job.get("end_date"))
                    if start and end and start > end:
                        impossible_flags.append({
                            "id":     cid,
                            "reason": f"role at '{company}' has start_date {start} > end_date {end}",
                            "detail": job
                        })
                        break

            # Flag open_to_work=True but last_active > 1 year ago AND rrr < 0.05
            if len(impossible_flags) < 5:
                if (otw is True
                        and dsa is not None and dsa > 365
                        and rrr is not None and float(rrr) < 0.05):
                    impossible_flags.append({
                        "id":     cid,
                        "reason": (f"open_to_work=True but last active {dsa}d ago "
                                   f"and recruiter_response_rate={rrr:.2f}"),
                        "detail": {
                            "last_active_date": signals.get("last_active_date"),
                            "recruiter_response_rate": rrr,
                            "open_to_work_flag": otw
                        }
                    })

        if total % 10_000 == 0:
            print(f"  ... {total:,} records processed", file=sys.stderr)


# ── pretty-print ──────────────────────────────────────────────────────────────

SEP  = "=" * 70
SEP2 = "-" * 70

def bar(count, total, width=30):
    frac = count / total if total else 0
    filled = int(frac * width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {frac*100:5.1f}%"


print()
print(SEP)
print("  REDROB CANDIDATE DATASET  --  EXPLORATION SUMMARY")
print(f"  Total records: {total:,}                Date: {TODAY}")
print(SEP)

# ── 1. Top 30 titles ──────────────────────────────────────────────────────────
print()
print("1.  DISTRIBUTION OF CURRENT TITLE  (top 30)")
print(SEP2)
top30 = title_counter.most_common(30)
max_count = top30[0][1] if top30 else 1
for rank, (title, cnt) in enumerate(top30, 1):
    pct = cnt / total * 100
    print(f"  {rank:2d}. {title:<40s}  {cnt:6,}  ({pct:5.2f}%)")

# ── 2. Years of experience histogram ─────────────────────────────────────────
print()
print("2.  YEARS OF EXPERIENCE  (histogram)")
print(SEP2)
ordered_buckets = [yoe_label(lo) for lo, hi in zip(YOE_EDGES, YOE_EDGES[1:])] + ["50+"]
for bucket in ordered_buckets:
    cnt = yoe_buckets.get(bucket, 0)
    print(f"  {bucket:<8s}  {cnt:6,}  {bar(cnt, total)}")

# ── 3. Geography ──────────────────────────────────────────────────────────────
print()
print("3.  GEOGRAPHY")
print(SEP2)
total_known = india_count + non_india_count
print(f"  India      : {india_count:6,}  {bar(india_count, total_known)}")
print(f"  Elsewhere  : {non_india_count:6,}  {bar(non_india_count, total_known)}")
print()
print("  Key Indian cities (from location field):")
for city in sorted(INDIA_CITIES):
    cnt = city_counter.get(city, 0)
    print(f"    {city.capitalize():<12s}  {cnt:6,}  {bar(cnt, india_count)}")

# ── 4. Behavioral signals ─────────────────────────────────────────────────────
print()
print("4.  BEHAVIORAL SIGNALS")
print(SEP2)

print()
print("  a) recruiter_response_rate  (0.0 - 1.0, buckets of 0.1)")
for bucket in range(10):
    lo = bucket * 0.1
    hi = lo + 0.1
    cnt = rrr_hist.get(bucket, 0)
    print(f"    [{lo:.1f}-{hi:.1f})  {cnt:6,}  {bar(cnt, total)}")

print()
print("  b) last_active_date  (days since last seen)")
for label in ["<30d", "30-90d", "90-180d", "180-365d", ">365d"]:
    cnt = last_active_hist.get(label, 0)
    print(f"    {label:<12s}  {cnt:6,}  {bar(cnt, total)}")

print()
print("  c) open_to_work_flag")
for val in [True, False]:
    cnt = open_to_work_counter.get(val, 0)
    label = "True " if val else "False"
    print(f"    {label}  {cnt:6,}  {bar(cnt, total)}")

print()
print("  d) interview_completion_rate  (quintile buckets 0-1)")
quintile_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
for i, label in enumerate(quintile_labels):
    cnt = icr_hist.get(i, 0)
    print(f"    [{label}]  {cnt:6,}  {bar(cnt, total)}")

# ── 5. Impossible / suspicious profiles ───────────────────────────────────────
print()
print("5.  IMPOSSIBLE / SUSPICIOUS PROFILES  (first 5 examples)")
print(SEP2)
if not impossible_flags:
    print("  No impossible profiles found in this pass.")
else:
    for i, ex in enumerate(impossible_flags, 1):
        print(f"\n  Example {i}  --  {ex['id']}")
        print(f"    Reason : {ex['reason']}")
        detail = ex["detail"]
        if isinstance(detail, dict):
            for k, v in detail.items():
                print(f"    {k}: {v}")

print()
print(SEP)
print("  END OF SUMMARY")
print(SEP)
