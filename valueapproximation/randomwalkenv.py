import numpy as np

class RandomWalkEnvironment():
    def env_init(self, env_info={}):
        """
        Setup for the environment called when the experiment first starts.
        
        Set parameters needed to setup the 500-state random walk environment.
        
        Assume env_info dict contains:
        {
            num_states: 500 [int],
            start_state: 250 [int],
            left_terminal_state: 0 [int],
            right_terminal_state: 501 [int],
            seed: int
        }
        """
        
        # set random seed for each run

        self.rand_generator = np.random.RandomState(env_info.get("seed")) 
        
        # set each class attribute
        self.num_states = env_info["num_states"] 
        self.start_state = env_info["start_state"] 
        self.left_terminal_state = env_info["left_terminal_state"] 
        self.right_terminal_state = env_info["right_terminal_state"]

    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state from the environment.
        """

        # set self.reward_state_term tuple
        reward = 0.0
        state = self.start_state
        is_terminal = False
                
        self.reward_state_term = (reward, state, is_terminal)
        
        # return first state from the environment
        return self.reward_state_term[1]
        
    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state,
                and boolean indicating if it's terminal.
        """
        
        last_state = self.reward_state_term[1]
        
        # set reward, current_state, and is_terminal
        #
        # action: specifies direction of movement - 0 (indicating left) or 1 (indicating right)  [int]
        # current state: next state after taking action from the last state [int]
        # reward: -1 if terminated left, 1 if terminated right, 0 otherwise [float]
        # is_terminal: indicates whether the episode terminated [boolean]
        #
        # Given action (direction of movement), determine how much to move in that direction from last_state
        # All transitions beyond the terminal state are absorbed into the terminal state.
        
        if action == 0: # left
            current_state = max(self.left_terminal_state, last_state + self.rand_generator.choice(range(-100,0)))
        elif action == 1: # right
            current_state = min(self.right_terminal_state, last_state + self.rand_generator.choice(range(1,101)))
        else: 
            raise ValueError("Wrong action value")
        
        # terminate left
        if current_state == self.left_terminal_state: 
            reward = -1.0
            is_terminal = True
        
        # terminate right
        elif current_state == self.right_terminal_state:
            reward = 1.0
            is_terminal = True
        
        else:
            reward = 0.0
            is_terminal = False
        
        self.reward_state_term = (reward, current_state, is_terminal)
        
        return self.reward_state_term