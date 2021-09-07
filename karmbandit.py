# %%
import numpy as np
import matplotlib.pyplot as plt


class KArmNormalBandit:
    def __init__(self, means, variances):
        assert len(means) == len(
            variances), "length of mean and variance arrays don't match"
        self.means = means
        self.variances = variances
        self.l = len(means)
        self.rng = np.random.default_rng()
        #print(f"True variances are\n{self.variances}")
        #print(f"True means are\n{self.means}")

    def draw(self, i):
        assert i < self.l, "i not in array"
        assert i >= 0, "i below zero"
        return self.rng.normal(self.means[i], self.variances[i])


class ELearner:
    def __init__(self, length, warmup, e, init, bandit):
        self.l = length
        self.e = e
        self.w = warmup
        self.n = np.zeros(length)
        self.bandit = bandit
        self.q = np.full(l, init)
        self.rng = np.random.default_rng()
        self.rewards = []

    def warmup(self):
        for i in range(self.w):
            a = i % l
            r = self.act(a)
            self.update(a, r)

    def act(self, a):
        r = self.bandit.draw(a)
        self.rewards.append(r)
        return r

    def update(self, a, r):
        self.n[a] += 1
        self.q[a] = self.q[a]+(r-self.q[a])/self.n[a]

    def select(self):
        if self.rng.random() > self.e:
            return np.argmax(self.q)
        else:
            return self.rng.integers(l)

    def run(self, iterations):
        self.warmup()
        for _ in range(iterations):
            a = self.select()
            r = self.act(a)
            self.update(a, r)
        return self.rewards


class GradientLearner:
    def __init__(self, length, alpha, bandit):
        self.l = length
        self.bandit = bandit
        self.h = np.zeros(self.l)
        self.t = 1
        self.alpha = alpha
        self.rhat = 0
        self.rng = np.random.default_rng()
        self.rewards = []
        self.averagereward = []

    def act(self, a):
        r = self.bandit.draw(a)
        self.rhat = self.rhat + (r - self.rhat)/self.t
        self.rewards.append(r)
        self.averagereward.append(self.rhat)
        return r

    def update(self, a, r):
        self.t += 1
        #print(f"r is {r}, rhat is {self.rhat}, diff is {r-self.rhat}")
        probs = np.exp(self.h)/sum(np.exp(self.h))

        old = self.h
        for i in range(self.l):
            if i == a:
                self.h[i] = self.h[i] + self.alpha * \
                    (r - self.rhat) * (1-probs[i])
            else:
                self.h[i] = self.h[i] - self.alpha * (r - self.rhat) * probs[i]
        # print(old-self.h)

    def select(self):
        probs = np.exp(self.h)/sum(np.exp(self.h))

        a = self.rng.choice(self.l, p=probs)
        return a

    def run(self, iterations):
        for _ in range(iterations):
            a = self.select()
            r = self.act(a)
            self.update(a, r)
        return self.rewards


# %%
ntrials = 100
nsteps = 1000
warmup = 0
l = 10
rng = np.random.default_rng()
results = np.zeros((ntrials, nsteps))

for trial in range(ntrials):
    means = rng.random(10)*5+5  # generate numbers from 0 to 10
    variances = rng.random(10)
    bandit = KArmNormalBandit(means, variances)
    learner = GradientLearner(l,  0.1, bandit)
    results[trial, :] = learner.run(nsteps)[warmup:]
avg_result = np.mean(results, axis=0)

plt.plot(avg_result, label="ucb learner")
plt.legend()

# %%
