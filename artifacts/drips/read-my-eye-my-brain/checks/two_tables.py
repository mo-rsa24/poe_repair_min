"""Answer key for abstract sentence 2: the same model at two clinics.

Clinic A is the object already built. Clinic B is invented the same way: 100
patients, 20 with disease, but the model catches fewer of them -- the picture
"may not generalise reliably" would predict.
"""
from step1_patients import patients as clinic_a
from count_table import count_table


def sens_spec(patients):
    caught, missed, false_alarm, cleared = count_table(patients)
    return caught / (caught + missed), cleared / (cleared + false_alarm)


clinic_b = (
    [(1, 1)] * 12   # has disease, model refers
    + [(1, 0)] * 8  # has disease, model misses
    + [(0, 1)] * 8  # healthy, model refers anyway
    + [(0, 0)] * 72 # healthy, model lets go
)

if __name__ == "__main__":
    sa, spa = sens_spec(clinic_a)
    sb, spb = sens_spec(clinic_b)
    print(f"clinic A: sens {sa} spec {spa}")
    print(f"clinic B: sens {sb} spec {spb}")
    print(f"gap: {sa - sb:.2f}")
