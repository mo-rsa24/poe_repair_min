"""Answer key, step 2: the list counted into a two-by-two table."""
from step1_patients import patients


def count_table(patients):
    caught = sum(1 for e, m in patients if e == 1 and m == 1)
    missed = sum(1 for e, m in patients if e == 1 and m == 0)
    false_alarm = sum(1 for e, m in patients if e == 0 and m == 1)
    cleared = sum(1 for e, m in patients if e == 0 and m == 0)
    return caught, missed, false_alarm, cleared


if __name__ == "__main__":
    print(*count_table(patients))
