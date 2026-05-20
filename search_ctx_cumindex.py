"""
Search the local CTX cumindex to find correct product IDs,
then auto-update config.yaml with verified real IDs.

Run from your project root:
  python search_ctx_cumindex.py
"""
import sys
import yaml
from pathlib import Path

cumindex_path = Path("data/raw/ctx/.cache/ctx_cumindex.csv")
config_path   = Path("config.yaml")

if not cumindex_path.exists():
    print(f"ERROR: {cumindex_path} not found.")
    print("Run: python main.py --step download   (to download the cumindex first)")
    sys.exit(1)

# Site definitions: name -> list of lat/lon substrings to search for
# CTX encodes lat/lon in the product ID as e.g. "36S230W"
# We search multiple nearby variants to handle rounding
SITES = {
    "gasa": {
        "desc": "Gasa Crater (35.68S, 230.72W)",
        "terms": ["36S230W", "35S231W", "36S229W", "35S230W", "1440"],
        "filter": lambda pid: any(x in pid for x in ["36S230", "35S230", "35S231"]),
    },
    "palikir": {
        "desc": "Palikir Crater (41.6S, 202.3W)",
        "terms": ["42S202W", "41S202W", "42S203W", "41S203W", "1380"],
        "filter": lambda pid: any(x in pid for x in ["42S202", "41S202", "42S203"]),
    },
    "russell": {
        "desc": "Russell Crater (54.3S, 347.3E = 12.7W) - note: CTX uses W longitudes",
        # Russell is at 12.7W = 347.3E. CTX product IDs use West longitude.
        # So search for "54S347W", "55S347W", "54S348W" etc.
        "terms": ["54S347W", "55S347W", "54S348W", "55S348W", "54S346W"],
        "filter": lambda pid: any(x in pid.upper() for x in
                                  ["54S347", "55S347", "54S348", "55S348", "54S346"]),
    },
}

print(f"Searching {cumindex_path} ({cumindex_path.stat().st_size/1e6:.1f} MB)...")
print("This takes about 3-5 seconds...\n")

results = {site: [] for site in SITES}

with open(cumindex_path, encoding="latin-1", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 2:
            continue
        vol = parts[0]
        file_path = parts[1]
        fname = file_path.split("/")[-1].split("\\")[-1]
        pid = fname.replace(".IMG", "").replace(".img", "")
        pid_upper = pid.upper()

        for site, cfg in SITES.items():
            if cfg["filter"](pid_upper):
                results[site].append((vol.lower(), pid))

print("=" * 65)
print("RESULTS: CTX Products Near Each Study Site")
print("=" * 65)

final_ids = {}
for site, cfg in SITES.items():
    hits = results[site]
    print(f"\n{cfg['desc']}  ({len(hits)} products found)")
    print("-" * 60)
    if not hits:
        print("  [NONE FOUND] - check coordinate encoding in search_ctx_cumindex.py")
        final_ids[site] = []
    else:
        # Sort by product ID (chronological approximation)
        hits_sorted = sorted(hits, key=lambda x: x[1])
        for vol, pid in hits_sorted[:10]:
            url = f"https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/{vol}/data/{pid}.IMG"
            print(f"  {pid}")
            print(f"    URL: {url}")
        if len(hits) > 10:
            print(f"  ... and {len(hits)-10} more")
        final_ids[site] = [pid for _, pid in hits_sorted[:6]]

# Auto-update config.yaml
if config_path.exists() and any(final_ids.values()):
    print("\n" + "=" * 65)
    print("Updating config.yaml with verified CTX product IDs...")
    with open(config_path) as f:
        content = f.read()

    # Build new known_images block
    lines = []
    lines.append("    known_images:")
    for site, pids in final_ids.items():
        lines.append(f"      {site}:")
        if pids:
            for pid in pids:
                lines.append(f'        - "{pid}"')
        else:
            lines.append(f'        []  # No products found near this site')
    new_block = "\n".join(lines) + "\n"

    # Find and replace
    import re
    # Match from "    known_images:" to next top-level ctx key or next sensor
    pattern = r'    known_images:.*?(?=\n  \w|\Z)'
    # Find CTX section
    ctx_start = content.find("  ctx:")
    if ctx_start == -1:
        print("Could not find 'ctx:' in config.yaml")
    else:
        ctx_section = content[ctx_start:]
        km_start = ctx_section.find("    known_images:")
        if km_start != -1:
            abs_start = ctx_start + km_start
            # Find end of known_images block (next key at 2-space indent)
            rest = content[abs_start + 16:]  # skip "    known_images:"
            end_match = re.search(r'\n  \w', rest)
            abs_end = abs_start + 16 + (end_match.start() if end_match else len(rest))
            content = content[:abs_start] + new_block + content[abs_end:]
            with open(config_path, "w") as f:
                f.write(content)
            print("[OK] config.yaml updated with verified CTX IDs")
        else:
            print("Could not find known_images block in CTX section")
else:
    print("\nNo config.yaml update (file not found or no results)")

print("\nDone. Now run: python main.py --step download")