from collections import Counter
import math

from data import train_data, val_data, vocab_size


def empirical_bigram_entropy(data):
    pairs = Counter()
    counts = Counter()

    for index in range(len(data) - 1):
        current = data[index].item()
        following = data[index + 1].item()
        pairs[(current, following)] += 1
        counts[current] += 1

    total = sum(counts.values())
    entropy = 0.0
    for current, current_count in counts.items():
        conditional = 0.0
        for following in range(vocab_size):
            probability = pairs.get((current, following), 0) / current_count
            if probability:
                conditional -= probability * math.log(probability)
        entropy += (current_count / total) * conditional
    return entropy


train_entropy = empirical_bigram_entropy(train_data)
validation_entropy = empirical_bigram_entropy(val_data)

print(f"empirical training bigram entropy: {train_entropy:.4f}")
print(f"empirical validation bigram entropy: {validation_entropy:.4f}")
print(
    "Interpretation: compare the training value with an averaged training loss. "
    "The validation value describes a separately estimated held-out distribution; "
    "neither should be compared with one randomly sampled batch."
)
