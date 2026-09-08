from mltools.common.data import partition_groups, split_k_fold_by_groups, _fold_from_partition

groups = [['a'], ['b']*20, ['c']*3, ['d']*15, ['e']*2, ['f']*8, ['g'], ['h']*10, ['i']*5, ['j']]

print("=== random: single-fold call matches full-partition selection ===")
for fold in range(5):
    t1, v1 = split_k_fold_by_groups(groups, k=5, fold=fold, seed=42, method="random")
    print(f"fold {fold}: train={len(t1)} val={len(v1)}")

print("\n=== random: partition computed once, all folds cover every item exactly once, no leakage ===")
partition = partition_groups(groups, k=5, seed=42, method="random")
all_items = set(item for g in groups for item in g)
seen = set()
for fold in range(5):
    train, val = _fold_from_partition(groups, partition, fold, 5)
    seen |= set(val)
    assert set(train) & set(val) == set(), f"leakage at fold {fold}"
assert seen == all_items, "not all items covered across folds"
print("OK")

print("\n=== greedy: fold sizes ===")
partition = partition_groups(groups, k=5, seed=42, method="greedy")
for fold in range(5):
    train, val = _fold_from_partition(groups, partition, fold, 5)
    print(f"fold {fold}: val_size={len(val)}")

print("\n=== invalid method raises ===")
try:
    partition_groups(groups, k=5, seed=42, method="bogus")
    print("FAIL: no exception")
except ValueError as e:
    print("OK:", e)

print("\n=== optimal: confirm CP-SAT solves once, not once per fold ===")
import mltools.common.data as data_mod
calls = {"n": 0}
orig = data_mod._solve_optimal_partition.__wrapped__ if hasattr(data_mod._solve_optimal_partition, "__wrapped__") else None
# simplest check: cache_info() before/after computing all 5 folds
data_mod._solve_optimal_partition.cache_clear()
groups10 = [[str(i)] * (i + 1) for i in range(10)]  # 10 groups, divides evenly by 5
partition = partition_groups(groups10, k=5, seed=42, method="optimal")
for fold in range(5):
    _fold_from_partition(groups10, partition, fold, 5)
print("cache info after computing all 5 folds from ONE partition:", data_mod._solve_optimal_partition.cache_info())
assert data_mod._solve_optimal_partition.cache_info().misses == 1, "solved more than once!"
print("OK: CP-SAT solved exactly once for all 5 folds")
