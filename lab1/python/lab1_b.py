import random
import monkdata as m
import dtree as d


def partition(data, fraction, seed=0):
    """
    Shuffle data and split into two parts:
    - first part: fraction of data
    - second part: the rest
    """
    ldata = list(data)
    random.Random(seed).shuffle(ldata)
    break_point = int(len(ldata) * fraction)
    return ldata[:break_point], ldata[break_point:]


def prune_complete(tree, validation_data, verbose=True):
    """
    Reduced error pruning:
    Repeatedly:
      - generate all trees that are ONE prune smaller (allPruned)
      - pick the one with best validation accuracy
      - stop when no candidate is better than current
    """
    current = tree
    current_acc = d.check(current, validation_data)

    step = 0
    if verbose:
        print(f"Initial validation accuracy: {current_acc:.4f}")

    while True:
        step += 1

        # 1) generate all trees that are 1 prune smaller
        candidates = d.allPruned(current)  # tuple of trees

        # 2) find best candidate on validation
        best_tree = current
        best_acc = current_acc

        for t in candidates:
            acc = d.check(t, validation_data)
            if acc > best_acc:
                best_acc = acc
                best_tree = t

        if verbose:
            print(f"Step {step}: tried {len(candidates)} prunes, best val acc = {best_acc:.4f}")

        # 3) stopping rule: if nothing improved, stop
        if best_tree is current:
            if verbose:
                print("Stop: no prune improved validation accuracy.")
            break

        # 4) accept improvement and continue pruning from that tree
        current = best_tree
        current_acc = best_acc

    return current


def evaluate(name, full_train_data, test_data, train_fraction=0.6, seed=0, verbose=True):
    """
    For one MONK dataset:
      - split full_train_data -> train + val
      - build full tree on train
      - prune using val
      - report accuracy on train / val / test for full and pruned trees
    """
    train_data, val_data = partition(full_train_data, train_fraction, seed=seed)

    # Build full tree from the TRAIN part only (not using validation!)
    full_tree = d.buildTree(train_data, m.attributes)

    # Prune using validation
    pruned_tree = prune_complete(full_tree, val_data, verbose=verbose)

    # Accuracies
    acc_train_full = d.check(full_tree, train_data)
    acc_val_full = d.check(full_tree, val_data)
    acc_test_full = d.check(full_tree, test_data)

    acc_train_pruned = d.check(pruned_tree, train_data)
    acc_val_pruned = d.check(pruned_tree, val_data)
    acc_test_pruned = d.check(pruned_tree, test_data)

    print("\n" + "=" * 40)
    print(f"{name}  (train_fraction={train_fraction}, seed={seed})")
    print("-" * 40)
    print(f"FULL  : train={acc_train_full:.4f}  val={acc_val_full:.4f}  test={acc_test_full:.4f}")
    print(f"PRUNED: train={acc_train_pruned:.4f}  val={acc_val_pruned:.4f}  test={acc_test_pruned:.4f}")
    print("=" * 40)

    return {
        "name": name,
        "full": (acc_train_full, acc_val_full, acc_test_full),
        "pruned": (acc_train_pruned, acc_val_pruned, acc_test_pruned),
    }


def main():
    # Run for all 3 MONK datasets
    datasets = [
        ("MONK-1", m.monk1, m.monk1test),
        ("MONK-2", m.monk2, m.monk2test),
        ("MONK-3", m.monk3, m.monk3test),
    ]

    # You can change these
    train_fraction = 0.6
    seed = 0
    verbose = True

    for name, train_data, test_data in datasets:
        evaluate(name, train_data, test_data, train_fraction=train_fraction, seed=seed, verbose=verbose)


if __name__ == "__main__":
    main()
