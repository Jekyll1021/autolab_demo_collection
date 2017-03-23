from env_wrapper import *

class Agent:
    """a templete for MDP agent"""
    def ___init___(self, environment, goal = 0):
        """parameters:
        --environment: as name indicates
        --state: same
        --time: a time counter that accounts for logical times the current agent takes 
        --next_state: store [observation, reward, done, info] of the previous action.
        --goal: a time that all action ends.
        """
        self.environment = environment
        self.state = self.environment.state
        self.time = 0
        self.next_state = None
        self.goal = goal

    def get_action_space(self):
        # TODO: get self.environment.action_space after action_space is implemented
        pass

    def get_action(self):
        # TO BE OVERRIDEN for MDP learning purposes.
        pass

    def take_action(self, action):
        """it now takes in an action (list/tuple of [position, rotation]), 
        execute the action(move arm to indicated pos, rot)
        and returns:"""
        """
        observation: object --- an environment-specific object representing your observation of the environment; 
            in this env new list of pos, rot
        reward: float --- amount of reward achieved by the previous action. 
            The scale varies between environments, and can also be customized using BLANK method.
        done: boolean --- whether it's time to reset the environment again. 
            Most tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. 
        info: dict --- value TBD, naive thoughts are to include info for next step
        """
        # TO BE OVERRIDEN for MDP learning purposes.
        return self.environment.step(action)

    def get_reward(self):
        """call this function after taking an action to find reward for the previous action."""
        if self.next_state:
            return self.next_state[1]
        else:
            return 0

    def execute(self):
        # TO BE OVERRIDEN for MDP learning purposes.
        """method that execute a series of action that is learn by this MDP agent."""
        pass

    def learn(self, i):
        # TO BE OVERRIDEN for MDP learning purposes.
        """method that makes the agent explore and learn i times."""
        pass

class SequenceAgent(Agent):
    """An agent that takes in a series of agents and execute their specified actions in order."""
    def __init__(self, agents_list, environment, goal = 0):
        Agent.___init___(self, environment, goal)
        self.agents = agents_list
        self.goal = 0
        for agent in self.agents:
            if self.environment != agent.environment:
                raise ValueError("Agent cannot execute in current environment!")

    def execute(self):
        for agent in self.agents:
            agent.execute()
            