"""Analyze GPU kernel trace to understand UNKNOWN kernel overhead."""

import json
from collections import defaultdict

TRACE = "/p/scratch/westai0047/nanochat/logs/flash_atlas_trace/plugins/profile/2026_04_03_09_35_18/jrc0910.trace.json"

with open(TRACE) as f:
    data = json.load(f)

events = data["traceEvents"]

# Filter to GPU kernel events (ph='X' = complete events, on GPU tid)
# Exclude Python-level events (names starting with $) and command_buffer events
gpu_kernels = [
    e for e in events
    if e.get("dur", 0) > 0
    and e.get("ph") == "X"
    and not e["name"].startswith("$")
    and e["name"] not in ("command_buffer::update", "command_buffer::execute")
]

print("=" * 80)
print("TRACE ANALYSIS: GPU Kernel Events")
print("=" * 80)

# 1. Total kernel launches
print(f"\n1. Total GPU kernel launches (dur > 0): {len(gpu_kernels)}")
total_time_us = sum(e["dur"] for e in gpu_kernels)
print(f"   Total GPU kernel time: {total_time_us:.0f} us = {total_time_us/1000:.1f} ms")

# Separate UNKNOWN vs named
unknown = [e for e in gpu_kernels if e["name"] == "<UNKNOWN>"]
named = [e for e in gpu_kernels if e["name"] != "<UNKNOWN>"]

unknown_time = sum(e["dur"] for e in unknown)
named_time = sum(e["dur"] for e in named)

print(f"\n   UNKNOWN kernels: {len(unknown)} ({len(unknown)/len(gpu_kernels)*100:.1f}%)")
print(f"   UNKNOWN time:    {unknown_time:.0f} us = {unknown_time/1000:.1f} ms ({unknown_time/total_time_us*100:.1f}%)")
print(f"   Named kernels:   {len(named)} ({len(named)/len(gpu_kernels)*100:.1f}%)")
print(f"   Named time:      {named_time:.0f} us = {named_time/1000:.1f} ms ({named_time/total_time_us*100:.1f}%)")

# 2. UNKNOWN duration distribution
print("\n" + "=" * 80)
print("2. UNKNOWN kernels by duration range")
print("=" * 80)

ranges = [
    ("< 1 us", 0, 1),
    ("1 - 10 us", 1, 10),
    ("10 - 100 us", 10, 100),
    ("100 us - 1 ms", 100, 1000),
    ("1 - 10 ms", 1000, 10000),
    ("> 10 ms", 10000, float("inf")),
]

print(f"\n{'Range':<18} {'Count':>8} {'Total (us)':>12} {'Total (ms)':>12} {'% of UNKNOWN':>14}")
print("-" * 70)
for label, lo, hi in ranges:
    in_range = [e for e in unknown if lo <= e["dur"] < hi]
    t = sum(e["dur"] for e in in_range)
    pct = t / unknown_time * 100 if unknown_time > 0 else 0
    print(f"{label:<18} {len(in_range):>8} {t:>12.1f} {t/1000:>12.3f} {pct:>13.1f}%")

avg_dur = unknown_time / len(unknown) if unknown else 0
median_dur = sorted(e["dur"] for e in unknown)[len(unknown) // 2] if unknown else 0
print(f"\n   Average UNKNOWN kernel duration: {avg_dur:.2f} us")
print(f"   Median  UNKNOWN kernel duration: {median_dur:.2f} us")

# 3. Launch overhead estimate
print("\n" + "=" * 80)
print("3. Launch overhead estimate")
print("=" * 80)

LAUNCH_OVERHEAD_US = 5  # conservative: 5 us per kernel launch
overhead_unknown = len(unknown) * LAUNCH_OVERHEAD_US
overhead_all = len(gpu_kernels) * LAUNCH_OVERHEAD_US

print(f"\n   Assuming ~{LAUNCH_OVERHEAD_US} us launch overhead per kernel:")
print(f"   UNKNOWN kernels: {len(unknown)} launches x {LAUNCH_OVERHEAD_US} us = {overhead_unknown:.0f} us = {overhead_unknown/1000:.1f} ms")
print(f"   All kernels:     {len(gpu_kernels)} launches x {LAUNCH_OVERHEAD_US} us = {overhead_all:.0f} us = {overhead_all/1000:.1f} ms")
print(f"\n   If all {len(unknown)} UNKNOWN kernels fused into 1:")
print(f"     Saved launch overhead: {(len(unknown)-1) * LAUNCH_OVERHEAD_US / 1000:.1f} ms")
print(f"     That's {(len(unknown)-1) * LAUNCH_OVERHEAD_US / unknown_time * 100:.1f}% of UNKNOWN time")

# Also estimate with 10us overhead
LAUNCH_OVERHEAD_US_HIGH = 10
overhead_high = len(unknown) * LAUNCH_OVERHEAD_US_HIGH
print(f"\n   With {LAUNCH_OVERHEAD_US_HIGH} us launch overhead estimate:")
print(f"     UNKNOWN launch overhead: {overhead_high/1000:.1f} ms ({overhead_high/unknown_time*100:.1f}% of UNKNOWN time)")

# 4. Top 20 longest UNKNOWN kernels
print("\n" + "=" * 80)
print("4. Top 20 UNKNOWN kernels by individual duration")
print("=" * 80)

unknown_sorted = sorted(unknown, key=lambda e: -e["dur"])
print(f"\n{'Rank':<6} {'Duration (us)':>14} {'Duration (ms)':>14} {'Timestamp (us)':>16}")
print("-" * 55)
for i, e in enumerate(unknown_sorted[:20]):
    print(f"{i+1:<6} {e['dur']:>14.2f} {e['dur']/1000:>14.4f} {e['ts']:>16.1f}")

top20_time = sum(e["dur"] for e in unknown_sorted[:20])
print(f"\n   Top 20 UNKNOWN total: {top20_time:.0f} us = {top20_time/1000:.1f} ms ({top20_time/unknown_time*100:.1f}% of UNKNOWN time)")

# 5. Fraction of tiny kernels
print("\n" + "=" * 80)
print("5. Launch-overhead-dominated kernels (< 10 us)")
print("=" * 80)

tiny_all = [e for e in gpu_kernels if e["dur"] < 10]
tiny_unknown = [e for e in unknown if e["dur"] < 10]

print(f"\n   All kernels < 10 us: {len(tiny_all)} / {len(gpu_kernels)} = {len(tiny_all)/len(gpu_kernels)*100:.1f}%")
print(f"   UNKNOWN   < 10 us:  {len(tiny_unknown)} / {len(unknown)} = {len(tiny_unknown)/len(unknown)*100:.1f}%")

tiny_time = sum(e["dur"] for e in tiny_all)
tiny_unknown_time = sum(e["dur"] for e in tiny_unknown)
print(f"\n   Time in tiny (<10us) kernels: {tiny_time/1000:.1f} ms ({tiny_time/total_time_us*100:.1f}% of total)")
print(f"   Time in tiny UNKNOWN:         {tiny_unknown_time/1000:.1f} ms ({tiny_unknown_time/unknown_time*100:.1f}% of UNKNOWN)")

# 6. Summary verdict
print("\n" + "=" * 80)
print("6. VERDICT")
print("=" * 80)

large_unknown = [e for e in unknown if e["dur"] >= 100]
large_unknown_time = sum(e["dur"] for e in large_unknown)
small_unknown = [e for e in unknown if e["dur"] < 100]
small_unknown_time = sum(e["dur"] for e in small_unknown)

print(f"\n   UNKNOWN >= 100 us: {len(large_unknown)} kernels, {large_unknown_time/1000:.1f} ms ({large_unknown_time/unknown_time*100:.1f}% of UNKNOWN time)")
print(f"   UNKNOWN <  100 us: {len(small_unknown)} kernels, {small_unknown_time/1000:.1f} ms ({small_unknown_time/unknown_time*100:.1f}% of UNKNOWN time)")

if large_unknown_time > unknown_time * 0.5:
    print("\n   -> UNKNOWN time is dominated by a FEW LARGE kernels doing real work.")
    print("      Fusion won't help much; need to identify and optimize these large kernels.")
else:
    print("\n   -> UNKNOWN time is dominated by MANY SMALL kernels (launch overhead).")
    print("      Kernel fusion could significantly reduce overhead.")

# 7. Bonus: named kernel summary
print("\n" + "=" * 80)
print("7. Named kernel summary (top 15 by total time)")
print("=" * 80)

name_stats = defaultdict(lambda: {"count": 0, "total_us": 0})
for e in named:
    # Truncate long names
    name = e["name"][:80]
    name_stats[name]["count"] += 1
    name_stats[name]["total_us"] += e["dur"]

sorted_names = sorted(name_stats.items(), key=lambda x: -x[1]["total_us"])
print(f"\n{'Name':<82} {'Count':>6} {'Total (ms)':>10} {'Avg (us)':>10}")
print("-" * 115)
for name, s in sorted_names[:15]:
    avg = s["total_us"] / s["count"]
    print(f"{name:<82} {s['count']:>6} {s['total_us']/1000:>10.1f} {avg:>10.1f}")
