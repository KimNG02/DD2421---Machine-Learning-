import random
import math
import monkdata as m
import dtree as d
import matplotlib.pyplot as plt

# -------------------------
# Helper: split data
# -------------------------
def partition(data, fraction, seed=None):
    ldata = list(data)
    if seed is not None:
        random.seed(seed)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

# -------------------------
# Complete pruning:
# repeatedly try all 1-step prunes, pick best on validation, stop if no improvement
# -------------------------
def prune_complete(tree, validation_data):
    current = tree
    current_acc = d.check(current, validation_data)

    while True:
        candidates = d.allPruned(current)  # all trees 1 prune step away

        best_tree = current
        best_acc = current_acc

        for t in candidates:
            acc = d.check(t, validation_data)
            if acc > best_acc:
                best_acc = acc
                best_tree = t

        # stop if no candidate improved validation accuracy
        if best_tree is current:
            break

        current = best_tree
        current_acc = best_acc

    return current

# -------------------------
# Run experiment for one dataset and one fraction, repeated many times
# returns list of test errors from each run
# -------------------------
def run_fraction_experiment(dataset_train, dataset_test, fraction, runs=50):
    test_errors = []

    for seed in range(runs):
        train_set, val_set = partition(dataset_train, fraction, seed=seed)

        full_tree = d.buildTree(train_set, m.attributes)
        pruned_tree = prune_complete(full_tree, val_set)

        test_acc = d.check(pruned_tree, dataset_test)
        test_error = 1.0 - test_acc
        test_errors.append(test_error)

    return test_errors

# -------------------------
# Main: fractions to test
# -------------------------
fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
runs = 50

results = {
    "MONK-1": [],
    "MONK-3": []
}

# MONK-1
for f in fractions:
    errs = run_fraction_experiment(m.monk1, m.monk1test, f, runs=runs)
    mean_err = sum(errs) / len(errs)
    std_err = math.sqrt(sum((e - mean_err) ** 2 for e in errs) / len(errs))
    results["MONK-1"].append((mean_err, std_err))

# MONK-3
for f in fractions:
    errs = run_fraction_experiment(m.monk3, m.monk3test, f, runs=runs)
    mean_err = sum(errs) / len(errs)
    std_err = math.sqrt(sum((e - mean_err) ** 2 for e in errs) / len(errs))
    results["MONK-3"].append((mean_err, std_err))

# -------------------------
# Print table of results
# -------------------------
print("\n=== Assignment 7 results (mean ± std test error over {} runs) ===".format(runs))
print("fraction   MONK-1_test_error      MONK-3_test_error")
for i, f in enumerate(fractions):
    m1_mean, m1_std = results["MONK-1"][i]
    m3_mean, m3_std = results["MONK-3"][i]
    print(f"{f:0.1f}      {m1_mean:0.4f} ± {m1_std:0.4f}      {m3_mean:0.4f} ± {m3_std:0.4f}")

# -------------------------
# Plot
# -------------------------
m1_means = [x[0] for x in results["MONK-1"]]
m1_stds  = [x[1] for x in results["MONK-1"]]

m3_means = [x[0] for x in results["MONK-3"]]
m3_stds  = [x[1] for x in results["MONK-3"]]

plt.figure()
plt.errorbar(fractions, m1_means, yerr=m1_stds, marker='o', label="MONK-1")
plt.errorbar(fractions, m3_means, yerr=m3_stds, marker='o', label="MONK-3")

plt.xlabel("fraction (train split size)")
plt.ylabel("test error (1 - accuracy)")
plt.title("Test error vs fraction (mean ± std over runs)")
plt.legend()
plt.grid(True)
plt.show()
