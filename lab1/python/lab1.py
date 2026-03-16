import monkdata as m
import dtree as d

###Assignment 1
#compute the entropy of the dataset MONK
#entropy is a number that measures how mixed the labels are
print ("MONK-1 entropy:", d.entropy(m.monk1))
print ("MONK-2 entropy:", d.entropy(m.monk2))
print ("MONK-3 entropy:", d.entropy(m.monk3))

###Assignment 3
datasets = [ #a big list
    ("MONK-1", m.monk1), #(name, dataset)
    ("MONK-2", m.monk2),
    ("MONK-3", m.monk3),
]

for name, dataset in datasets: #goes through the list one tuple at a time
    print("\n" + name)
    information_gains = []
    for i, attr in enumerate(m.attributes):
        g = d.averageGain(dataset, attr) #calculate information gain for that attribute
        information_gains.append(g) #add the value to the list
        print(f"a{i+1}: {g:.6f}") #f-string, print g as a float with 6 decimals 


###Assignment 5a
#Step 1: Root choice - from assignment 3
root = m.attributes[4] #A5 is index 4
print("\nRoot attribute for MONK-1 should be:", root)

#Step 2: Split MONK-1 by the root attribute
subsets = {v: d.select(m.monk1, root, v) for v in root.values}

print("\nSubset sizes after splitting by A5:")
for v in root.values:
    print(f"A5={v}: {len(subsets[v])} samples, entropy={d.entropy(subsets[v]):.6f}")

#Step 3: For each subset, compute best attribute for next level
remaining = [a for a in m.attributes if a != root]

best_next = {} #store best attribute for each branch A5=value

print("\nBest next attribute for each branch (A5=value):")
for v, subset in subsets.items():
    #If subset is pure (entropy=0), no need to split more
    if d.entropy(subset) == 0:
        best_next[v] = None
        print(f"A5={v}: PURE subset -> leaf (all same class)")
        continue

    gains = [(d.averageGain(subset, a), a) for a in remaining]
    gains_sorted = sorted(gains, key=lambda x: x[0], reverse=True)

    print(f"\nA5={v}:")
    for g, a in gains_sorted:
        print(f"  gain({a}) = {g:.6f}")

    best_next[v] = gains_sorted[0][1]
    print("Best next attribute:", best_next[v])

#Step 4: Build the 2-level tree rules + majority leaf labels
print("\n\n2-level TREE (manual) with majority leaves")

for v, subset in subsets.items():
    #If pure, label directly
    if d.entropy(subset) == 0:
        label = d.mostCommon(subset) #True/False
        sign = "+" if label else "-"
        print(f"If A5={v} -> {sign} (pure)")
        continue
    
    a2 = best_next[v]
    print(f"If A5={v} then test {a2}:")

    for v2 in a2.values:
        subset2 = d.select(subset, a2, v2)
        if len(subset2) == 0:
            #if no data, fall back to majority of the parent subset
            label = d.mostCommon(subset)
        else:
            label = d.mostCommon(subset2)

        sign = "+" if label else "-"
        print(f"   If {a2}={v2} -> {sign}  (size={len(subset2)})")


#Step 5: Compare with ID3 routine buildTree at depth 2
print("\n\nID3 buildTree with maxdepth=2")
t2 = d.buildTree(m.monk1, m.attributes, maxdepth=2)
print(t2)


###Assignment 5b
datasets = [
    ("MONK-1", m.monk1, m.monk1test),
    ("MONK-2", m.monk2, m.monk2test),
    ("MONK-3", m.monk3, m.monk3test),

]
print("Dataset   acc_train  acc_test   E_train   E_test")
for name, train, test in datasets:
    t = d.buildTree(train, m.attributes) #full tree
    acc_train = d.check(t, train)
    acc_test = d.check(t, test)
    E_train = 1 - acc_train
    E_test = 1 - acc_test

    print(f"{name:7}  {acc_train:.4f}     {acc_test:.4f}   {E_train:.4f}   {E_test:.4f}")
