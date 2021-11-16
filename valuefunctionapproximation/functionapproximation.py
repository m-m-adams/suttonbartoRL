import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from randomwalkenv import RandomWalkEnvironment
from TDagent import TDAgent


agent_info = {"num_states": 500,
              "num_hidden_layer": 1,
              "num_hidden_units": 100,
              "step_size": 0.1,
              "discount_factor": 1.0,
              "beta_m": 0.9,
              "beta_v": 0.99,
              "epsilon": 0.0001,
              "seed": 10
              }


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(500, 100)  # 5*5 from image dimension
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        # If the size is a square, you can specify with a single number
        x = self.fc2(x)
        return x


# Suppose state = 250
state = 250
valuenetwork = Net()
test_agent = TDAgent()
test_agent.agent_init(valuenetwork, agent_info)
test_agent.agent_start(state)

# Define function to run experiment


def run_experiment(environment, agent, environment_parameters, value_function, agent_parameters, experiment_parameters):
    environment = environment()
    agent = agent()
    # save learned state value at the end of each run
    agent_state_val = torch.zeros((experiment_parameters["num_runs"],
                                   environment_parameters["num_states"]))

    env_info = {"num_states": environment_parameters["num_states"],
                "start_state": environment_parameters["start_state"],
                "left_terminal_state": environment_parameters["left_terminal_state"],
                "right_terminal_state": environment_parameters["right_terminal_state"]}

    agent_info = {"num_states": environment_parameters["num_states"],
                  "step_size": agent_parameters["step_size"],
                  "discount_factor": environment_parameters["discount_factor"],
                  "beta_m": agent_parameters["beta_m"],
                  "beta_v": agent_parameters["beta_v"],
                  "epsilon": agent_parameters["epsilon"]
                  }

    print('Setting - Neural Network with 100 hidden units')

    # one agent setting
    for run in tqdm(range(1, experiment_parameters["num_runs"]+1)):

        for episode in range(1, experiment_parameters["num_episodes"]+1):
            env_info["seed"] = run+episode
            agent_info["seed"] = run+episode
            environment.env_init(env_info)
            agent.agent_init(value_function, agent_info)

            terminal = False
            state = environment.env_start()
            agent.agent_start(state)
            while not terminal:

                action = agent.agent_policy(state)

                (reward, state, terminal) = environment.env_step(action)
                #if terminal: print(reward, state)

                agent.agent_step(reward, state)
            agent.agent_end(reward)

            if episode % experiment_parameters["episode_eval_frequency"] == 0:
                current_V = agent.agent_message("get state value")
            # if last episode
            elif episode == experiment_parameters["num_episodes"]:
                current_V = agent.agent_message("get state value")
        print(agent.agent_message("get state value"))


# Experiment parameters
experiment_parameters = {
    "num_runs": 1,
    "num_episodes": 500,
    "episode_eval_frequency": 100  # evaluate every 10 episode
}

# Environment parameters
environment_parameters = {
    "num_states": 500,
    "start_state": 250,
    "left_terminal_state": 0,
    "right_terminal_state": 500,
    "discount_factor": 1.0
}

# Agent parameters
agent_parameters = {
    "num_hidden_layer": 1,
    "num_hidden_units": 100,
    "step_size": 0.01,
    "beta_m": 0.9,
    "beta_v": 0.999,
    "epsilon": 0.0001,
}

current_env = RandomWalkEnvironment
current_agent = TDAgent

# run experiment
run_experiment(current_env, current_agent, environment_parameters,
               valuenetwork, agent_parameters, experiment_parameters)
