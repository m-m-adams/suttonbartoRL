import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from randomwalkenv import RandomWalkEnvironment
from TDagent import TDAgent


def run_experiment(environment, agent, environment_parameters, agent_parameters, experiment_parameters):
    environment = environment()
    agent = agent()

    value_runs = np.zeros(
        [experiment_parameters["num_runs"], environment_parameters["num_states"]])

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
    for run in tqdm(range(experiment_parameters["num_runs"])):

        env_info["seed"] = run
        agent_info["seed"] = run
        environment.env_init(env_info)
        agent.agent_init(agent_info)
        for episode in range(1, experiment_parameters["num_episodes"]+1):

            terminal = False
            state = environment.env_start()
            agent.agent_start(state)
            action = agent.agent_policy(state)
            (reward, state, terminal) = environment.env_step(action)
            while not terminal:
                agent.agent_step(reward, state)
                action = agent.agent_policy(state)

                (reward, state, terminal) = environment.env_step(action)
                # if terminal: print(reward, state)

            agent.agent_end(reward)

            if episode == experiment_parameters["num_episodes"]:
                current_V = agent.agent_message("get state value")
        value_runs[run, :] = current_V
    plt.plot(value_runs.mean(axis=0))
    plt.show()


# Experiment parameters
experiment_parameters = {
    "num_runs": 20,
    "num_episodes": 1000,
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
               agent_parameters, experiment_parameters)
