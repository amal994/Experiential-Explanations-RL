import gym


class SpecialRewardsWrapper(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        reward = env._reward()
        agent_pos = env.agent_pos
        cur_object = env.grid.get(agent_pos[0], agent_pos[1])
        if cur_object:
            if cur_object.type == "ball":
                if cur_object.color == env.bad:
                    done = True
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
