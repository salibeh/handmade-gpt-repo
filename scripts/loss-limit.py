from data import vocab_size, train_data
from collections import Counter
import math

pairs = Counter()
counts = Counter()
for i in range(len(train_data) - 1):
    c1 = train_data[i].item()
    c2 = train_data[i+1].item()
    pairs[(c1, c2)] += 1
    counts[c1] += 1

total_entropy = 0.0
total_count = sum(counts.values())

for c1 in counts:
    c1_count = counts[c1]
    entropy_c1 = 0.0
    for c2 in range(vocab_size):
        p = pairs.get((c1, c2), 0) / c1_count
        if p > 0:
            entropy_c1 -= p * math.log(p)
    weight = c1_count / total_count
    total_entropy += weight * entropy_c1

print(f"Theoretical minimum loss (entropy floor): {total_entropy:.4f}")
