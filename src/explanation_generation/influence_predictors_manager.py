import os

import numpy as np
import torch

from src.rl_minigrid.utils.DQN import DQN


class InfluencePredictorsManager:
    """
    Influence predictor manager contains all the functions that deal directly with influence predictors,
    especially when it relates to using them together.
    """

    def __init__(self, device, model_dir, bad_influence_model_name, goal_influence_model_name,
                 other_influence_model_name=None):
        """
        Initialize the influence predictors manager:
        """
        self.device = device
        self.bad_policy = InfluencePredictor('negative', os.path.join(model_dir, bad_influence_model_name), self.device)
        self.goal_policy = InfluencePredictor('positive', os.path.join(model_dir, goal_influence_model_name),
                                              self.device)
        #self.other_policy = InfluencePredictor('negative', os.path.join(model_dir, other_influence_model_name),
        #                                       self.device)
    def get_obs_from_location(self, env, i, di): #TODO this function shouldn't be in this class
        env.place_agent(i, (1, 1), rand_dir=False)
        env.agent_dir = di
        obs = env.gen_obs()
        img = env.get_obs_render(obs['image'], tile_size=8)
        return obs, img

    def get_qs_of_obs(self, env, i, di, agent):
        """
        Given an environment and the agent location + direction. Get all the Q values from the influence predictors.
        :params
        env: the environment object (minigrid)
        i: agent location
        di: agent direction
        """
        obs, img = self.get_obs_from_location(env, i, di)
        dqn_state = self.prepare_state(img)
        bad_q = max(self.bad_policy.predict(dqn_state)).item()
        goal_q = max(self.goal_policy.predict(dqn_state)).item()
        #other_q = max(self.other_policy.predict(dqn_state)).item()
        action, q = agent.get_action_and_values(obs)  # TODO: Move this?

        return obs, action, q, bad_q, goal_q#, other_q

    def prepare_state(self, state):
        """
        [...]
        :params
        :returns
        """
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state).float()
        state = state.unsqueeze(0).to(self.device)
        return state
    def get_action_of_obs(self, env, i, di, agent, threshold=-float("inf")):
        obs, img = self.get_obs_from_location(env, i, di)
        dqn_state = self.prepare_state(img)
        bad_q = self.bad_policy.predict(dqn_state)
        goal_q = self.goal_policy.predict(dqn_state)
        max_action, action = self.get_action_from_influence_predictors(obs, agent, goal_q, bad_q, threshold)
        return obs, max_action, action
    # Influence predictor helper function
    def get_action_from_influence_predictors(self, obs, agent, goal_influence, bad_influence, threshold=-float("inf")):
        """For faithfulness measure"""
        #g_max_action = np.argmax(goal_influence.cpu().detach().numpy())
        #b_max_action = np.argmax(bad_influence.cpu().detach().numpy())
        if max(goal_influence) < threshold and max(bad_influence) < threshold :
            return None
        sum_q = goal_influence - bad_influence
        max_action = np.argmax(sum_q.cpu().detach().numpy())
        action, _ = agent.get_action_and_values(obs)

        return max_action, action
    def get_action_from_one_influence_predictor(self, obs, agent, influence, threshold=-float("inf")):
        """For faithfulness measure"""
        if max(influence) < threshold:
            return None
        g_max_action = np.argmax(influence.cpu().detach().numpy())
        action, _ = agent.get_action_and_values(obs)
        return g_max_action, action

    def get_raw_influences_for_state(self, img):
        """ For faithfulness measure"""
        dqn_state = self.prepare_state(img)
        goal_influence = self.goal_policy.predict(dqn_state)
        bad_influence = self.bad_policy.predict(dqn_state)
        return goal_influence, bad_influence
    # Influence predictor helper function
    def get_action_predictions_scene(self, env, all_pos, agent, threshold, multiple=True, influence="goal"):
        """For faithfulness measure"""
        env.reset()
        agent_actions = {}
        influence_actions = {}
        goal_influences = {}
        bad_influences = {}
        for i in all_pos:
            di = 1
            if env.grid.get(*i) is None:
                env.place_agent(i, (1, 1), rand_dir=False)
                env.agent_dir = di
                obs = env.gen_obs()
                img = env.get_obs_render(obs['image'], tile_size=8)
                goal_influence, bad_influence = self.get_raw_influences_for_state(img)
                if multiple:
                    returned_action = self.get_action_from_influence_predictors(obs, agent, goal_influence, bad_influence, threshold)
                elif influence == "goal":
                    returned_action = self.get_action_from_one_influence_predictor(obs, agent, goal_influence, threshold)
                elif influence == "bad":
                    returned_action = self.get_action_from_one_influence_predictor(obs, agent, bad_influence, threshold)
                if returned_action:
                    agent_actions[i] = returned_action[1]
                    influence_actions[i] = returned_action[0]
                goal_influences[i] = goal_influence
                bad_influences[i] = bad_influence
        return agent_actions, influence_actions, goal_influences, bad_influences


class InfluencePredictor:
    """
    A class for loading influence predictors and handling their outcomes.
    """

    def __init__(self, influence_type, path, device, screen_height=7, screen_width=7, num_actions=7, num_reward_classes=1):
        self.influence = influence_type  # Positive or Negative.
        self.device = device
        p_state_dict = torch.load(path, map_location=self.device)  # Load influence predictor.
        self.predictor = DQN(screen_height, screen_width, num_actions, num_reward_classes).to(device)
        self.predictor.load_state_dict(p_state_dict)

    def predict(self, state):
        return self.predictor(state)[0]
