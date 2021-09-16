# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

MAX_CARS = 10


def calcv(state, action, state_value):
    """
    @state: [# of cars in first location, # of cars in second location]
    @action: selected action
    @stateValue: state value matrix
    """
    if (action > state[0]) or (-state[1] > action):
        # print("impossible")
        return -1000
    # initailize total return
    returns = 0.0

    # cost for moving cars
    returns -= 2*abs(action)

    # moving cars - n1st is first lot, n2nd is second lot
    n1st = min(state[0] - action, MAX_CARS)
    n2nd = min(state[1] + action, MAX_CARS)
    assert n1st >= 0
    assert n2nd >= 0

    # #deterministic evaluation
    # valid1st = min(n1st, 3)
    # valid2nd = min(n2nd, 4)
    # reward = 10*(valid1st+valid2nd)
    # n1st -= valid1st
    # n2nd -= valid2nd
    # returns += 1 * \
    #     (reward + 0.9 *
    #         state_value[min(n1st+3, MAX_CARS), min(n2nd+2, MAX_CARS)])
    rentals = 0
    # go through all possible rental requests. Arbitrarily cut poisson at 10 to save time
    for req1 in range(0, 10):
        for req2 in range(0, 10):
            # probability for current combination of rental requests
            prob_request = poisson.pmf(req1, 2) * poisson.pmf(req2, 4)

            # valid rental requests should be less than actual # of cars
            valid1st = min(n1st, req1)
            valid2nd = min(n2nd, req2)

            # get credits for renting
            reward = 10*(valid1st+valid2nd)
            n1st_req = n1st - valid1st
            n2nd_req = n2nd - valid2nd

            # for ret1 in range(2, 4):
            #     for ret2 in range(1, 3):
            #         prob_return = poisson.pmf(ret1, 3) * poisson.pmf(ret2, 2)
            #         n1st = min(n1st + ret1, 20)
            #         n2nd = min(n2nd + ret2, 20)
            #         prob = prob_return * prob_request
            rentals += prob_request * \
                (reward + 0.9 *
                 state_value[min(n1st_req+3, MAX_CARS), min(n2nd_req+1, MAX_CARS)])

    #print(rentals, returns)
    return returns + rentals


class PolicyIterator:
    def __init__(self, threshold):
        # set value for all states to 0
        self.v = np.zeros((MAX_CARS+1, MAX_CARS+1))
        # deterministic, init all policies to move no cars
        self.p = np.zeros((MAX_CARS+1, MAX_CARS+1), dtype=int)

        self.threshold = threshold

    def evaluate(self):
        print("evaluating")
        delta = 10
        while delta > self.threshold:
            for n1 in range(0, MAX_CARS+1):
                for n2 in range(0, MAX_CARS+1):
                    v = self.v[n1, n2]
                    vnext = calcv((n1, n2), self.p[n1, n2], self.v)
                    self.v[n1, n2] = vnext
                    delta = abs(v-vnext)

    def improve(self):
        print("improving")
        stable = True
        for n1 in range(MAX_CARS+1):
            for n2 in range(MAX_CARS+1):
                aold = self.p[n1, n2]
                v = -1000.0
                anext = aold
                for a in range(-5, 6):
                    vnew = calcv((n1, n2), a, self.v)
                    #print(f"value of moving {a} from {n1} to {n2} is {vnew}")

                    if vnew > v:
                        v = vnew
                        anext = a
                if anext != aold:
                    self.p[n1, n2] = anext
                    stable = False
        return stable

    def run(self):
        self.evaluate()
        while not self.improve():
            print("policy")
            print(np.rot90(self.p, axes=(1, 0)))
            self.evaluate()

        return self.v, self.p


learner = PolicyIterator(0.01)
a = learner.run()
print(np.rot90(a[1], axes=(1, 0)).T)

plt.imshow(np.rot90(a[0], axes=(1, 0)).T)

# %%
