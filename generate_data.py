import random
import pandas as pd

def save_to_csv(data, filename):
    df = pd.DataFrame(data, columns=['sequence', 'label'])
    df.to_csv(filename, index=False)
    print(f"Saved: {filename} ({len(data)} samples)")

# 1. Regular: Parity (Even number of 1s)
def gen_parity(n=10000, max_len=50):
    data = []
    for _ in range(n):
        length = random.randint(10, max_len)
        seq = "".join(random.choice("01") for _ in range(length))
        label = 1 if seq.count("1") % 2 == 0 else 0
        data.append((seq, label))
    return data

# 2. Context-Free: Dyck-1 (Balanced Parentheses)
def gen_dyck(n=10000, max_len=50):
    data = []
    while len(data) < n:
        # Generate random combinations of ()
        length = random.randint(5, max_len // 2) * 2
        # Simple algorithm to generate balanced parens
        seq = []
        balance = 0
        for _ in range(length):
            if balance == 0:
                seq.append("(")
                balance += 1
            elif balance >= (length - len(seq)):
                seq.append(")")
                balance -= 1
            else:
                choice = random.choice(["(", ")"])
                seq.append(choice)
                balance += 1 if choice == "(" else -1
        data.append(("".join(seq), 1))
        # Add a negative sample (unbalanced)
        bad_seq = list("".join(seq))
        idx = random.randint(0, len(bad_seq)-1)
        bad_seq[idx] = ")" if bad_seq[idx] == "(" else "("
        data.append(("".join(bad_seq), 0))
    return data[:n]

# 3. Context-Sensitive: a^n b^n c^n
def gen_abc(n=10000, max_len=60):
    data = []
    while len(data) < n:
        n_val = random.randint(2, max_len // 3)
        # Positive case
        data.append(("a"*n_val + "b"*n_val + "c"*n_val, 1))
        # Negative case (mismatched counts)
        offsets = [random.randint(-1, 1) for _ in range(3)]
        if offsets == [0, 0, 0]: offsets[0] = 1 # Force a mismatch
        data.append(("a"*(n_val+offsets[0]) + "b"*(n_val+offsets[1]) + "c"*(n_val+offsets[2]), 0))
    return data[:n]

if __name__ == "__main__":
    save_to_csv(gen_parity(), "data_parity.csv")
    save_to_csv(gen_dyck(), "data_dyck.csv")
    save_to_csv(gen_abc(), "data_abc.csv")