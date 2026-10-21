"""Example for step 3: the two extreme models on the same clinic day.

The expert's grades stay fixed. Only the model's call changes.
"""
from step1_patients import patients
from count_table import count_table

def sens_spec(patients):
    caught, missed, false_alarm, cleared = count_table(patients)
    return caught / (caught + missed), cleared / (cleared + false_alarm)

refer_everyone = [(expert, 1) for expert, model in patients]
refer_nobody = [(expert, 0) for expert, model in patients]

for name, day in [("refer everyone", refer_everyone), ("your model", patients), ("refer nobody", refer_nobody)]:
    print(f"{name:15}", *count_table(day), " ", *sens_spec(day))
