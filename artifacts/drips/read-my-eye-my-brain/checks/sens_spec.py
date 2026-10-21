"""Answer key, stage 1: the two fractions read off the table."""
from step1_patients import patients
from count_table import count_table

caught, missed, false_alarm, cleared = count_table(patients)
sensitivity = caught / (caught + missed)          # of those with disease, the share the model refers
specificity = cleared / (cleared + false_alarm)   # of the healthy, the share the model lets go

if __name__ == "__main__":
    print(sensitivity, specificity)
