import torch

import src.rl_minigrid.utils as utils
from src.rl_minigrid.model import ACModel
import random

class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 device=None, argmax=False, num_envs=1, use_memory=False, use_text=False):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        self.acmodel = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text)
        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=self.device)

        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(self.device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])

    def get_action_and_values(self, obs, random_action=None, argmax=True):
        """
        A modified action function that returns the chosen action and its value.
        :params
        obs: the state object instance of TODO: add classname.
        random_action: whether to return a completeley random action.
        argmax: whether to return actions based on argmax function
        :returns
        action: a number representing the agent's action.
        value: the value the agent associates with the state
        """

        # This is the get_actions method from the agent.
        process_obs = self.preprocess_obss([obs], device=self.device)
        dist, value, self.memories = self.acmodel(process_obs, self.memories)

        if random_action:
            action = random_action.value
        elif argmax:
            action = dist.probs.max(1, keepdim=True)[1][0].item()
        else:
            # When the model is well-trained, the probs of getting the correct action are 90% +.
            # Therefore, sample, would perform just like max almost all the time.
            # action = dist.sample()[0].item()
            #action = torch.topk(dist.probs, 2)[1][0][1].item()
            action = random.randint(0, dist.probs.shape[1]-2)

        return action, value.item()