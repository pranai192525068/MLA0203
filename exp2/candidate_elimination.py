# Candidate-Elimination Algorithm
# MLA02 - Fundamentals of Machine Learning

import csv


# -----------------------------------------
# Step 1: Read training data from CSV
# -----------------------------------------

training_data = []

with open("training_data.csv", "r") as file:
    reader = csv.reader(file)

    # Read header
    header = next(reader)

    # Read remaining data
    for row in reader:
        training_data.append(row)


# -----------------------------------------
# Step 2: Display training dataset
# -----------------------------------------

print("=" * 80)
print("                         TRAINING DATASET")
print("=" * 80)

print(f"{'Sky':<12}{'AirTemp':<12}{'Humidity':<12}"
      f"{'Wind':<12}{'Water':<12}{'Forecast':<12}{'EnjoySport':<12}")

print("-" * 80)

for data in training_data:
    print(f"{data[0]:<12}{data[1]:<12}{data[2]:<12}"
          f"{data[3]:<12}{data[4]:<12}{data[5]:<12}{data[6]:<12}")

print("=" * 80)


# -----------------------------------------
# Step 3: Initialize S and G
# -----------------------------------------

num_attributes = len(header) - 1

# Most specific hypothesis
S = ['Ø'] * num_attributes

# Most general hypothesis
G = [['?'] * num_attributes]


print("\nInitial Specific Boundary (S):")
print(S)

print("\nInitial General Boundary (G):")
print(G)


# -----------------------------------------
# Step 4: Candidate-Elimination Algorithm
# -----------------------------------------

for example in training_data:

    attributes = example[:-1]
    target = example[-1]

    # -------------------------------------
    # Positive example
    # -------------------------------------
    if target == 'Yes':

        # Remove inconsistent hypotheses from G
        G = [
            g for g in G
            if all(
                g[i] == '?' or g[i] == attributes[i]
                for i in range(num_attributes)
            )
        ]

        # Generalize S
        for i in range(num_attributes):

            if S[i] == 'Ø':
                S[i] = attributes[i]

            elif S[i] != attributes[i]:
                S[i] = '?'

    # -------------------------------------
    # Negative example
    # -------------------------------------
    else:

        # Generate specializations of G
        new_G = []

        for g in G:

            # If G is completely general
            if all(value == '?' for value in g):

                for i in range(num_attributes):

                    if S[i] != 'Ø':
                        specialized = g.copy()
                        specialized[i] = S[i]
                        new_G.append(specialized)

            else:
                new_G.append(g)

        G = new_G

    print("\nProcessed Example:")
    print(example)

    print("Specific Boundary (S):")
    print(S)

    print("General Boundary (G):")
    print(G)


# -----------------------------------------
# Step 5: Display final boundaries
# -----------------------------------------

print("\n" + "=" * 80)
print("                    FINAL VERSION SPACE")
print("=" * 80)

print("\nFinal Specific Boundary (S):")
print(S)

print("\nFinal General Boundary (G):")

for g in G:
    print(g)

print("\nCandidate-Elimination algorithm completed successfully.")