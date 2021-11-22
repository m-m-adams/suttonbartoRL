import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
import numpy as np


def one_hot(state, num_states):
    """
    Given num_state and a state, return the one-hot encoding of the state
    """
    # Create the one-hot encoding of state
    # one_hot_vector is a numpy array of shape (1, num_states)
    if state > num_states:
        print(state)

    one_hot_vector = torch.zeros((1, num_states)).float()
    one_hot_vector[0, int((state - 1))] = 1

    return one_hot_vector


class TDAgent():
    def __init__(self):
        self.name = "td_agent"
        pass

    class Net(nn.Module):

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(500, 100)  # 5*5 from image dimension
            self.fc2 = nn.Linear(100, 1)

        def forward(self, x):

            x = F.relu(self.fc1(x))
            # If the size is a square, you can specify with a single number
            x = self.fc2(x)
            return x

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD with a Neural Network.

        Assume agent_info dict contains:
        {
            step_size: float, 
            discount_factor: float,
            self.beta_m: float
            self.beta_v: float
            self.epsilon: float
            seed: int
        }
        """

        # Set random seed for weights initialization for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.valuenetwork = self.Net().to(self.device)

        self.optimizer = optim.Adam(
            self.valuenetwork.parameters(), lr=agent_info.get("step_size"))
        # Set random seed for policy for each run
        self.policy_rand_generator = np.random.RandomState(
            agent_info.get("seed"))

        self.num_states = agent_info.get("num_states")

        self.gamma = agent_info.get("discount_factor")

        self.last_state = None
        self.last_action = None

    def agent_policy(self, state):

        # Set chosen_action as 0 or 1 with equal probability.
        chosen_action = self.policy_rand_generator.choice([0, 1])
        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        # select action given state (using self.agent_policy()), and save current state and action (2 lines)
        # self.last_state = ?
        # self.last_action = ?

        ### START CODE HERE ###
        self.last_state = state
        self.last_action = self.agent_policy(state)
        ### END CODE HERE ###

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

        # Compute TD error (5 lines)
        # delta = None
        self.optimizer.zero_grad()
        ### START CODE HERE ###
        last_state = one_hot(self.last_state, self.num_states)
        v_last = self.valuenetwork(last_state.to(self.device))

        curr_state = one_hot(state, self.num_states)
        v = self.valuenetwork(curr_state.to(self.device))
        target = reward + self.gamma*v

        #print(v_last, target)
        L = nn.MSELoss()
        loss = L(v_last, target.detach())

        loss.backward()

        self.optimizer.step()

        # update self.last_state and self.last_action (2 lines)

        ### START CODE HERE ###
        self.last_state = state
        self.last_action = self.agent_policy(state)
        ### END CODE HERE ###

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # compute TD error (3 lines)
        # delta = None

        ### START CODE HERE ###
        last_state = one_hot(self.last_state, self.num_states)
        v_last = self.valuenetwork(last_state.to(self.device))

        L = nn.MSELoss()
        loss = L(v_last, torch.tensor([[reward]]).to(self.device))
        loss.backward()
        self.optimizer.step()

    def agent_message(self, message):
        if message == 'get state value':
            state_value = np.zeros(self.num_states)
            for state in range(1, self.num_states + 1):
                s = one_hot(state, self.num_states)
                state_value[state -
                            1] = self.valuenetwork(s.to(self.device))
            return state_value
