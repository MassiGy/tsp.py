import random

# Generate nodes with x, y in range [0, 100]
num_nodes = 100
nodes = [(round(random.uniform(0, 100), 1), round(random.uniform(0, 100), 1)) for _ in range(num_nodes)]

# Create file content
with open("./data/instance3.txt", "w") as f:
    f.write(f"{num_nodes}\n")
    for x, y in nodes:
        f.write(f"{x} {y}\n")

