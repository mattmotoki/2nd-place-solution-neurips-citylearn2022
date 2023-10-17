from gym.spaces import Box
from agents.user_agent import UserAgent


def dict_to_action_space(aspace_dict):
    return Box(
        low=aspace_dict["low"],
        high=aspace_dict["high"],
        dtype=aspace_dict["dtype"],
    )


class OrderEnforcingAgent:
    """
    Emulates order enforcing wrapper in Pettingzoo for easy integration
    Calls each agent step with agent in a loop and returns the action
    """

    def __init__(self):
        self.num_buildings = None
        self.action_space = None
        self.central_agent = UserAgent()

    def register_reset(self, obs_dict):
        """Get the first observation after env.reset, return action"""
        action_space = obs_dict["action_space"]
        self.action_space = [dict_to_action_space(asd) for asd in action_space]
        observation_list = obs_dict["observation"]
        self.num_buildings = len(observation_list)
        for agent_id, observation in enumerate(observation_list):
            action_space = self.action_space[agent_id]
            self.central_agent.set_action_space(agent_id, action_space)
            self.central_agent.reset_agent(agent_id)
        return self.compute_action(observation_list)

    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def compute_action(self, observation_list):
        return self.central_agent.compute_all_actions(observation_list)
