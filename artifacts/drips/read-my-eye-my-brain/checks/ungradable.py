"""Answer key for abstract sentence 5: photographs too poor to grade.

15 of the 100 photographs are invented as ungradable. Two ways to handle them:
drop them from the table, or count every one as a referral (the safe default
a real clinic would use, since an unreadable image cannot clear a patient).
"""
from step1_patients import patients as clinic_a


def sens_spec(patients):
    caught = sum(1 for e, m in patients if e == 1 and m == 1)
    missed = sum(1 for e, m in patients if e == 1 and m == 0)
    false_alarm = sum(1 for e, m in patients if e == 0 and m == 1)
    cleared = sum(1 for e, m in patients if e == 0 and m == 0)
    return caught / (caught + missed), cleared / (cleared + false_alarm)


# 15 ungradable, all pulled from the healthy block (3 false-alarm positions, 12
# cleared positions): no diseased patient's photograph is unreadable here, by
# choice, to isolate what the policy does to specificity alone.
ungradable_idx = set(range(20, 23)) | set(range(28, 40))  # 15 healthy positions

dropped = [p for i, p in enumerate(clinic_a) if i not in ungradable_idx]
referred = [p if i not in ungradable_idx else (p[0], 1) for i, p in enumerate(clinic_a)]

if __name__ == "__main__":
    print("dropped:", len(dropped), sens_spec(dropped))
    print("counted as referral:", len(referred), sens_spec(referred))
