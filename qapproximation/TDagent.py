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

    one_hot_vector = torch.zeros((num_states)).float()
    one_hot_vector[int((state - 1))] = 1

    return one_hot_vector


class TDAgent():
    def __init__(self):
        self.name = "td_agent"
        pass

    class Net(nn.Module):

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1+500, 100)  # 5*5 from image dimension
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
        self.beta = agent_info.get("beta_v")
        self.epsilon = agent_info.get("epsilon")
        self.action_space = torch.tensor([0, 1])
        self.meanR = 0
        self.last_state = None
        self.last_action = None

    def agent_policy(self, state):
        state_vec = one_hot(state, self.num_states)
        # Set chosen_action as 0 or 1 with equal probability.
        maxv=-np.Inf
        a=self.action_space[0]
        for action in self.action_space:
            features = torch.cat((action.unsqueeze(0), state_vec),0)
            v = self.valuenetwork(features.to(self.device))
            if v > maxv:
                maxv = v
                a = action
        r = self.policy_rand_generator.uniform(0,1)
        if r > self.epsilon:
            return a
        else:
            chosen_action = torch.tensor(self.policy_rand_generator.choice([0, 1]))
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
        #add the last action to the feature vector
        features = features = torch.cat((self.last_action.unsqueeze(0), last_state),0)
        v_last = self.valuenetwork(features.to(self.device))

        curr_state = one_hot(state, self.num_states)
        a = self.agent_policy(state)
        features = torch.cat((a.unsqueeze(0), curr_state),0)
        v = self.valuenetwork(features.to(self.device))

        delta = reward -self.meanR + v - v_last

        self.meanR = self.meanR + self.beta*reward

        #print(v_last, target)
        L = nn.MSELoss()
        loss = L(v_last, delta.detach())

        loss.backward()
        self.optimizer.step()

        # update self.last_state and self.last_action (2 lines)

        ### START CODE HERE ###
        self.last_state = state
        self.last_action = a
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
        features = features = torch.cat((self.last_action.unsqueeze(0), last_state),0)
        v_last = self.valuenetwork(features.to(self.device))

        L = nn.MSELoss()
        loss = L(v_last, torch.tensor([[reward]]).to(self.device))
        loss.backward()
        self.optimizer.step()

    def get_state_value(self):
        with torch.no_grad():
            state_value = np.zeros(self.num_states)
            state_action = np.zeros(self.num_states)
            for state in range(self.num_states):
                state_vec = one_hot(state, self.num_states)
                maxv = -np.Inf
                a=0
                for action in self.action_space:
                    features = torch.cat((action.unsqueeze(0), state_vec),0)
                    v = self.valuenetwork(features.to(self.device))
                    if v > maxv:
                        maxv = v
                        a = action
                state_value[state] = maxv.to("cpu").data.numpy()
                state_action[state] = a.to("cpu").data.numpy()
            return state_value, state_action

    
