# Capacity-pressure sweep

Book size fixed at the `medium` preset (14 vehicles, 140 requests); only the fleet's capacity/demand ratio changes. Each cell is the mean objective over seeds 11, 23, 37 (lower is better).

| capacity/demand | greedy | hungarian | regret | rl | rl vs regret |
|---|---|---|---|---|---|
| 1.50x | 1152 | 1607 | 723 | 919 | -27.0% |
| 1.35x | 1278 | 1758 | 746 | 917 | -22.9% |
| 1.20x | 1321 | 1963 | 1036 | 1229 | -18.7% |
| 1.10x | 1724 | 2078 | 1195 | 1284 | -7.4% |
| 1.00x | 2032 | 2445 | 1708 | 1633 | **+4.4%** |
| 0.90x | 2329 | 2675 | 3226 | 2059 | **+36.2%** |
| 0.85x | 2614 | 3180 | 4274 | 2626 | **+38.6%** |

The learned policy first overtakes the hand-tuned regret rule at **1.00x** capacity and leads by **38.6%** at the tightest point (0.85x).


Generated in 66.3 s by `make capacity` (`python -m app.capacity_sweep`).
