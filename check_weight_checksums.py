"""Check the regenerated Lean weight modules against WEIGHT_CHECKSUMS.txt (body after the header comment).

Usage: python3 check_weight_checksums.py
"""
import hashlib
import sys

ok = True
for line in open("WEIGHT_CHECKSUMS.txt"):
    if line.startswith("#") or not line.strip():
        continue
    want, path = line.split()
    try:
        s = open(path).read()
    except FileNotFoundError:
        print(f"MISSING  {path}")
        ok = False
        continue
    got = hashlib.sha256(s[s.index("-/"):].encode()).hexdigest()
    print(f"{'OK      ' if got == want else 'MISMATCH'} {path}")
    ok &= got == want
print("ALL WEIGHT MODULES MATCH" if ok else "WEIGHT MODULES DIFFER")
sys.exit(0 if ok else 1)
