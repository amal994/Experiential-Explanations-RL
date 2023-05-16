import matplotlib.pyplot as plt
import numpy as np

from .graph_viz import describe_observation
from .utils import load_obj


class SceneQVisualizer:

    def __init__(self, path):
        """Initialize and load all explanation graph information.
        Args:
        path: Path to the stored scene details. (Generated using observations_data_generation.py)
        """
        # 1. Load explanation graph and get env description
        """
        explanation_graph: instance of ExplanationGraph class contains the env object.
        env: instance of Environment (Minigrid) extracted from explanation_graph.
        width: width of environment
        height: height of environment
        all_pos: a list of tuples (x,y)
        """
        self.explanation_graph = load_obj(path, 'explanation_graph')
        self.env = self.explanation_graph.env
        self.width = self.env.width
        self.height = self.env.height

        # 2. Get all positions where the agent can be.
        self.all_pos = self.get_all_agent_positions()

    def get_scene_info(self):
        return self.explanation_graph, self.env, self.width, self.height, self.all_pos

    def get_all_agent_positions(self):
        all_pos = [(i, j) for i in range(self.width) for j in range(self.height)]
        for i in range(len(all_pos)):
            pos = all_pos[i]
            ij_object = self.env.grid.get(pos[0], pos[1])
            if ij_object:
                # If object is Key or Goal or Door:
                if ij_object.type not in {'goal', 'key', 'door', 'ball'}:  # TODO: Hard-coded change to minigrid call
                    # Remove infeasible locations from all_pos
                    all_pos[i] = None
        all_pos = set(all_pos)
        # Remove None object
        all_pos.discard(None)

        return all_pos

    def visualize_scene(self):
        """
        Visualize Environment scene
        Returns:
        """
        plt.rcParams["figure.figsize"] = [4.00, 4.00]
        plt.rcParams["figure.autolayout"] = True
        im = self.env.render(mode='rgb', tile_size=16)
        fig, ax = plt.subplots()
        ax.imshow(im, extent=[0, 7, 7, 0])
        plt.axis('equal')
        plt.show()

    def get_scene_q_values(self, ipm, agent):
        """
        Get all the Q values and observation descriptions of all states in a scene (env).
        :params
        explanation_graph: instance of ExplanationGraph class contains the env object.
        env: instance of Environment (Minigrid) extracted from explanation_graph.
        all_pos: a list of tuples (x,y)
        :returns
        obs_with_q: Dictionary containing the values from agent for every state.
        obs_with_bad_q: Dictionary containing the negative influence values for every state.
        obs_with_goal_q: Dictionary containing the positive influence values for every state.
        obs_with_other_q: Dictionary containing the other influence values for every state. (Other is a tiny continuos )
        obs_desc: Dictionary containing the observation descriptions (each observation description is a list of tuples (object_type, color))
        """
        self.env.reset()
        obs_with_q = {}
        obs_with_bad_q = {}
        obs_with_goal_q = {}
        #obs_with_other_q = {}
        obs_desc = {}
        for i in self.all_pos:
            di = 1
            if self.env.grid.get(*i) is None:
                obs, action, q, bad_q, goal_q = ipm.get_qs_of_obs(self.env, i, di, agent)
                obs_with_q[i] = q
                obs_with_bad_q[i] = bad_q
                obs_with_goal_q[i] = goal_q
                #obs_with_other_q[i] = other_q
                features = describe_observation(obs['image'])
                obs_desc[i] = [t[1] + " " + t[0] for t in features]  # Get object and color "Blue ball"

        return obs_with_q, obs_with_bad_q, obs_with_goal_q, obs_desc #obs_with_other_q,

    # Visualize Q values on the minigrid map.
    # requires a call to matplotlib as plt
    def visualize_qs(self, q_dictionary, cmap):
        """
        This function visualizes a dictionary of q values on a minigrid environment.
        It uses the minigrid environment render function and matplotlib libarary.
        :params
        all_pos: ?
        q_dictionary: a dictionary of q valeus for each point. {(x,y): q_value}
        height: height of minigrid environment
        width: width of minigrid environment
        explanation_graph: instance of ExplanationGraph class contains the env object.
        :returns
        void
        """
        all_positions = np.zeros([self.height, self.width])
        all_positions[all_positions == 0] = 'nan'
        for i in self.all_pos:
            try:
                all_positions[i[0]][i[1]] = q_dictionary[i]
            except KeyError:
                all_positions[i[0]][i[1]] = None
        plt.rcParams["figure.figsize"] = [4.00, 4.00]
        plt.rcParams["figure.autolayout"] = True
        im = self.explanation_graph.env.render(mode='rgb', tile_size=16)
        fig, ax = plt.subplots()
        ax.imshow(im, extent=[0, 7, 7, 0])
        plt.imshow(all_positions.T, cmap=cmap, extent=[0, 7, 7, 0])
        plt.axis('equal')
        plt.colorbar()
        plt.show()

    def get_features(self, obs_with_q, obs_with_bad_q, obs_with_goal_q, obs_desc):
        unique_objects = []
        for d in obs_desc:
            unique_objects.extend(obs_desc[d])
        unique_objects = list(set(unique_objects))
        print(unique_objects)
        feature_matrix = []
        feature_dict = []
        for d in obs_desc:
            processed_features = []  # Array of observation counts followed by q_values, then position, then direction.
            cnt_objects = [0] * len(unique_objects)
            for word in obs_desc[d]:
                cnt_objects[unique_objects.index(word)] += 1
            # processed_features.extend(cnt_objects)
            processed_features.append(obs_with_q[d])
            processed_features.append(obs_with_bad_q[d])
            processed_features.append(obs_with_goal_q[d])

            # processed_features.append(obs_with_other_q[d])

            feature_matrix.append(processed_features)
            feature_dict.append((d, processed_features))
        return feature_matrix, feature_dict

    def process_scene(self, ipm, agent, cmap='hot'):
        # Process scene and get feature matrix
        print("Initial Scene")
        self.visualize_scene()
        obs_with_q, obs_with_bad_q, obs_with_goal_q, obs_desc = self.get_scene_q_values(ipm, agent)
        feature_matrix, feature_dict = self.get_features(obs_with_q, obs_with_bad_q, obs_with_goal_q, obs_desc)

        # Visualize Original Q:
        print("Original Q")
        self.visualize_qs(obs_with_q, cmap)

        # Visualize bad Q:
        print("Bad Q")
        self.visualize_qs(obs_with_bad_q, cmap)

        # Visualize Goal Q:
        print("Goal Q")
        self.visualize_qs(obs_with_goal_q, cmap)

        # Visualize Other Q:
        #print("Other Q")
        #self.visualize_qs(obs_with_other_q)

        return self.env, feature_matrix, feature_dict