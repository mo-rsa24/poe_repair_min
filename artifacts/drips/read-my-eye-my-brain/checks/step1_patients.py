"""Answer key, step 1: one clinic day as a list of 100 (expert, model) pairs.

1 means disease worth referring (expert) or "refer" (model); 0 means not.
The counts are invented by the walk: the proposal gives no numbers.
"""
patients = (
    [(1, 1)] * 16    # has disease, model refers
    + [(1, 0)] * 4   # has disease, model misses
    + [(0, 1)] * 8   # healthy, model refers anyway
    + [(0, 0)] * 72  # healthy, model lets go
)

if __name__ == "__main__":
    n = len(patients)
    diseased = sum(expert for expert, model in patients)
    referred = sum(model for expert, model in patients)
    print(n, diseased, referred)
