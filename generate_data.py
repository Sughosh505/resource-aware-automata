import random
import pandas as pd

def save_to_csv(data, filename):
    # Added 'language_name' to match your 7-column requirements
    df = pd.DataFrame(data, columns=['sequence', 'label', 'language_name'])
    df.to_csv(filename, index=False)
    print(f"Saved: {filename} ({len(data)} samples)")

# 1. Regular: Parity (Label 0)
def gen_parity(n=10000, max_len=50):
    data = []
    for _ in range(n):
        length = random.randint(10, max_len)
        seq = "".join(random.choice("01") for _ in range(length))
        # Even if the parity is 'bad', it's still a Regular Language problem
        data.append((seq, 0, "parity")) 
    return data

# 2. Context-Free: Dyck-1 (Label 1)
def gen_dyck(n=10000, max_len=50):
    data = []
    while len(data) < n:
        length = random.randint(5, max_len // 2) * 2
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
        
        # Valid or invalid, balanced parentheses require a Stack (PDA)
        data.append(("".join(seq), 1, "dyck"))
        
        if len(data) < n:
            bad_seq = list("".join(seq))
            idx = random.randint(0, len(bad_seq)-1)
            bad_seq[idx] = ")" if bad_seq[idx] == "(" else "("
            data.append(("".join(bad_seq), 1, "dyck"))
    return data[:n]

# 3. Context-Sensitive: a^n b^n c^n (Label 2)
# 3. Context-Sensitive: a^n b^n c^n (Label 2)
def gen_abc(n=10000, max_len=60):
    data = []
    while len(data) < n:
        n_val = random.randint(5, max_len // 3)
        # Positive case (a^n b^n c^n) -> Label 2
        data.append(("a"*n_val + "b"*n_val + "c"*n_val, 2, "anbncn"))
        
        # Negative case (mismatched counts) -> Label 2
        # (It still takes a Turing Machine to prove it's wrong!)
        if len(data) < n:
            offsets = [random.randint(-1, 1) for _ in range(3)]
            if offsets == [0, 0, 0]: offsets[0] = 1 
            bad_seq = "a"*(n_val+offsets[0]) + "b"*(n_val+offsets[1]) + "c"*(n_val+offsets[2])
            data.append((bad_seq, 2, "anbncn"))
            
    return data[:n]

if __name__ == "__main__":
    save_to_csv(gen_parity(), "data_parity.csv")
    save_to_csv(gen_dyck(), "data_dyck.csv")
    save_to_csv(gen_abc(), "data_abc.csv")