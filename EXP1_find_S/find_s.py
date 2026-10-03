# FIND-S Algorithm
# MLA02 - Fundamentals of Machine Learning

import csv

# -----------------------------------------
# Step 1: Read training dataset from CSV
# -----------------------------------------

training_data = []

with open("training_data.csv", "r") as file:
    reader = csv.reader(file)

    # Skip the header row
    header = next(reader)

    # Read each row
    for row in reader:
        training_data.append(row)


# -----------------------------------------
# Step 2: Display the training dataset
# -----------------------------------------

print("=" * 75)
print("                    TRAINING DATASET")
print("=" * 75)

print(f"{'Sky':<12}{'AirTemp':<12}{'Humidity':<12}"
      f"{'Wind':<12}{'Water':<12}{'Forecast':<12}{'EnjoySport':<12}")

print("-" * 75)

for data in training_data:
    print(f"{data[0]:<12}{data[1]:<12}{data[2]:<12}"
          f"{data[3]:<12}{data[4]:<12}{data[5]:<12}{data[6]:<12}")

print("=" * 75)


# -----------------------------------------
# Step 3: Initialize hypothesis
# -----------------------------------------

hypothesis = ['Ø', 'Ø', 'Ø', 'Ø', 'Ø', 'Ø']

print("\nInitial Hypothesis:")
print(hypothesis)


# -----------------------------------------
# Step 4: Apply FIND-S Algorithm
# -----------------------------------------

for example in training_data:

    # Check whether the example is positive
    if example[-1] == 'Yes':

        # First positive example
        if hypothesis[0] == 'Ø':
            hypothesis = example[:-1]

        else:
            # Compare each attribute
            for i in range(len(hypothesis)):

                if hypothesis[i] != example[i]:
                    hypothesis[i] = '?'

        print("\nProcessed Example:")
        print(example)

        print("Current Hypothesis:")
        print(hypothesis)


# -----------------------------------------
# Step 5: Display final hypothesis
# -----------------------------------------

print("\n" + "=" * 75)
print("FINAL MOST SPECIFIC HYPOTHESIS")
print("=" * 75)

print(hypothesis)