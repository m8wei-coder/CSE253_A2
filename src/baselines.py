import math
import random
from collections import Counter, defaultdict


class NGramSampler:
    def __init__(self, n=1, seed=42):
        self.n = n
        self.rng = random.Random(seed)
        self.table = defaultdict(Counter)
        self.unigram = Counter()

    def fit(self, sequences):
        for seq in sequences:
            self.unigram.update(seq)
            padded = [None] * (self.n - 1) + list(seq)
            for i in range(self.n - 1, len(padded)):
                ctx = tuple(padded[i - self.n + 1 : i])
                self.table[ctx][padded[i]] += 1
        return self

    def _draw(self, counter):
        toks, weights = zip(*counter.items())
        return self.rng.choices(toks, weights=weights, k=1)[0]

    def sample(self, length, prefix=None):
        out = list(prefix or [])
        for _ in range(length):
            ctx = tuple(([None] * (self.n - 1) + out)[-(self.n - 1) :])
            counter = self.table.get(ctx) or self.unigram
            out.append(self._draw(counter))
        return out

    def token_logprob(self, ctx, token, alpha=1.0):
        """Add-alpha smoothed log p(token | ctx)."""
        counter = self.table.get(ctx)
        vocab_size = max(len(self.unigram), 1)
        if counter is None or sum(counter.values()) == 0:
            total = sum(self.unigram.values())
            prob = (self.unigram.get(token, 0) + alpha) / (total + alpha * vocab_size)
        else:
            total = sum(counter.values())
            prob = (counter.get(token, 0) + alpha) / (total + alpha * vocab_size)
        return math.log(prob)

    def nll(self, sequences, alpha=1.0):
        """Mean NLL per token in nats, plus perplexity."""
        total_logp, total_n = 0.0, 0
        for seq in sequences:
            padded = [None] * (self.n - 1) + list(seq)
            for i in range(self.n - 1, len(padded)):
                ctx = tuple(padded[i - self.n + 1 : i])
                total_logp += self.token_logprob(ctx, padded[i], alpha=alpha)
                total_n += 1
        mean_nll = -total_logp / max(total_n, 1)
        return mean_nll, math.exp(min(mean_nll, 50))


def random_atb(train_pairs, length, seed=42):
    rng = random.Random(seed)
    pool = []
    for _, atb in train_pairs:
        pool.extend(atb)
    return [rng.choice(pool) for _ in range(length)]


def nearest_neighbor_copy(train_pairs, soprano):
    best_pair, best_score = train_pairs[0], -1
    for s_train, atb in train_pairs:
        n = min(len(soprano), len(s_train))
        score = sum(int(a == b) for a, b in zip(soprano[:n], s_train[:n])) / max(n, 1)
        if score > best_score:
            best_pair, best_score = (s_train, atb), score
    return best_pair[1]


def rule_based_triad(soprano, id_to_token, token_to_id):
    out = []
    scale = {0, 2, 4, 5, 7, 9, 11}
    for sid in soprano:
        tok = id_to_token[int(sid)]
        if not tok.startswith("P_"):
            out.extend([token_to_id["<REST>"]] * 3)
            continue
        sp = int(tok.split("_", 1)[1])
        root_pc = max(scale, key=lambda pc: int((sp - pc) % 12 in (0, 4, 7)))
        root = sp - ((sp - root_pc) % 12)
        chord = [root + 4, root, root - 12]
        ranges = [(55, 74), (48, 69), (40, 64)]
        for p, (lo, hi) in zip(chord, ranges):
            while p < lo:
                p += 12
            while p > hi:
                p -= 12
            out.append(token_to_id.get(f"P_{p}", token_to_id["<REST>"]))
    return out
