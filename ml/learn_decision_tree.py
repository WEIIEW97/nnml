"""Write a Python function that implements the decision tree learning
algorithm for classification.

The function should use recursive binary splitting based on entropy and
information gain to build a decision tree. It should take a list of
examples (each example is a dict of attribute-value pairs) and a list of
attribute names as input, and return a nested dictionary representing
the decision tree.
"""

import math
from collections import Counter


def E(labels):
    label_counts = Counter(labels)
    total_count = len(labels)
    entropy = -sum(
        (count / total_count) * math.log2(count / total_count)
        for count in label_counts.values()
    )
    return entropy


def IG(examples, attr, target_attr):
    total_entropy = E([example[target_attr] for example in examples])
    values = set(example[attr] for example in examples)
    attr_entropy = 0
    for value in values:
        value_subset = [
            example[target_attr] for example in examples if example[attr] == value
        ]
        value_entropy = E(value_subset)
        attr_entropy += (len(value_subset) / len(examples)) * value_entropy
    return total_entropy - attr_entropy


def majority_class(examples, target_attr):
    return Counter([example[target_attr] for example in examples]).most_common(1)[0][0]


def learn_decision_tree(examples, attributes, target_attr):
    if not examples:
        return "No examples"
    if all(example[target_attr] == examples[0][target_attr] for example in examples):
        return examples[0][target_attr]
    if not attributes:
        return majority_class(examples, target_attr)

    gains = {attr: IG(examples, attr, target_attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    tree = {best_attr: {}}

    for value in set(example[best_attr] for example in examples):
        subset = [example for example in examples if example[best_attr] == value]
        new_attributes = attributes.copy()
        new_attributes.remove(best_attr)
        subtree = learn_decision_tree(subset, new_attributes, target_attr)
        tree[best_attr][value] = subtree

    return tree


if __name__ == "__main__":
    print(
        learn_decision_tree(
            [
                {"Outlook": "Sunny", "Wind": "Weak", "PlayTennis": "No"},
                {"Outlook": "Overcast", "Wind": "Strong", "PlayTennis": "Yes"},
                {"Outlook": "Rain", "Wind": "Weak", "PlayTennis": "Yes"},
                {"Outlook": "Sunny", "Wind": "Strong", "PlayTennis": "No"},
                {"Outlook": "Sunny", "Wind": "Weak", "PlayTennis": "Yes"},
                {"Outlook": "Overcast", "Wind": "Weak", "PlayTennis": "Yes"},
                {"Outlook": "Rain", "Wind": "Strong", "PlayTennis": "No"},
                {"Outlook": "Rain", "Wind": "Weak", "PlayTennis": "Yes"},
            ],
            ["Outlook", "Wind"],
            "PlayTennis",
        )
    )
