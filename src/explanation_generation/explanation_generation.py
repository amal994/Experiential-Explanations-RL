import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt

from src.explanation_generation.graph_viz import describe_observation
from src.explanation_generation.utils import load_obj

EXTRA_LENGTH = 20


class ExplanationGenerator:
    def __init__(self, path, ipm, agent):
        self.all_obss = load_obj(
            path,
            'all_obss')
        # A dictionary of observations.
        # For each observation, you can get their feature representations by calling
        # the method all_obs[obs_id].describe_observation()
        self.obss_seqs = load_obj(
            path,
            'sequential_obs')
        # A dictionary of array of tuples each tuple has an obs_id and the action that follows from this observation.
        self.explanation_graph = load_obj(path, 'explanation_graph')
        self.env = self.explanation_graph.env
        self.ipm = ipm
        self.agent = agent

    # Explanation generation => get Q values and their associated observation ids.
    def q_changes(self, path_name, sequences):
        # Ordered observation
        ordered_obs = []
        # Ordered Qs
        ordered_qs = []
        for j in sequences[path_name]['path']:
            if j[1]:
                ordered_obs.append(j[0])
                ordered_qs.append(j[2])

        assert len(ordered_obs) == len(ordered_qs)
        return ordered_obs, ordered_qs

    # Explanation generation => Get agent position for all possible
    def agent_poss(self, path_name, sequences):
        # All agent positions
        agent_positions = []
        for j in sequences[path_name]['path']:
            if j[1]:
                try:
                    # TODO: Potentially a bug in storing the positions in the sequence when creating the explanation_graph object.
                    agent_positions.append((j[4][0], j[4][1]))
                except TypeError as e:
                    agent_positions.append((j[3][0], j[3][1]))
        return agent_positions

    # Explanation generation => Get agent position for all possible
    def agent_dirs(self, path_name, sequences):
        # All agent directions
        agent_directions = []
        for j in sequences[path_name]['path']:
            if j[1]:
                try:
                    if j[4][0] and j[4][1]:  # STUPID BUG :: TODO : NEED TO FIX!
                        agent_directions.append(j[3])
                except TypeError as e:
                    agent_directions.append(j[4])
        return agent_directions

    # Explanation generation => Get agent position for all possible
    def compare_q_changes(self, path_num, sequneces):
        # Get paths with same prefix and initialize data structures
        paths = [key for key in sequneces if key.startswith(path_num)]
        q_paths = {p: [] for p in paths}
        obs_paths = {p: [] for p in paths}

        # Populate observations and Q changes.
        longest_sequence = 0
        append_obj = None
        for p in paths:
            # print(p)
            obs_paths[p], q_paths[p] = self.q_changes(p, sequneces)
            if len(obs_paths[p]) > longest_sequence:
                longest_sequence = len(obs_paths[p])

        # Unify lengths:
        for p in paths:
            r = longest_sequence - len(obs_paths[p])
            if r != 0:
                # Extend obs
                obs_paths[p].extend([append_obj] * r)
                # Extend Qs
                q_paths[p].extend([append_obj] * r)

        return obs_paths, q_paths

    # Explanation generation => Get agent position for all possible
    def get_average_q(self, path_qs):
        try:
            end_index = path_qs.index(None)
        except ValueError:
            end_index = len(path_qs)
        average = sum(path_qs[:end_index]) / (end_index - 1)
        return average

    # Explanation generation => Get agent position for all possible
    def measure_diff(self, main_t, counter_t):
        main_avg_q = self.get_average_q(main_t)
        counter_avg_q = self.get_average_q(counter_t)
        avg_q_diff = main_avg_q - counter_avg_q
        try:
            main_length = main_t.index(None)
        except ValueError:
            main_length = len(main_t)
        try:
            counter_length = counter_t.index(None)
        except ValueError:
            counter_length = len(counter_t)
        len_diff = main_length - counter_length
        return avg_q_diff, len_diff

    # Explanation generation => Get agent position for all possible
    def visualize_path(self, env, path, counter_path):
        markers = ['|', '_', 0, 1, 2, 3]
        plt.rcParams["figure.figsize"] = [4.00, 4.00]
        plt.rcParams["figure.autolayout"] = True
        im = env.render(mode='rgb', tile_size=16)
        fig, ax = plt.subplots()
        im = ax.imshow(im, extent=[0, 8, 8, 0])
        for i, j in path:
            plt.plot(i, j, marker=(5, 1), alpha=1, color="white")
        if counter_path:
            for x, y in counter_path:
                plt.plot(x, y, marker=(5, 1), alpha=1, color="red")
        plt.axis('equal')
        plt.show()

    # Explanation generation => (Phase 2)
    def change_point_detection(self, agent_q, bad_q, goal_q):
        stacked = np.array([agent_q, bad_q, goal_q])
        algo = rpt.Pelt(model="rbf").fit(stacked)
        result = algo.predict(pen=1)
        fig, ax = plt.subplots()
        ax.plot(agent_q, c="blue")
        ax.plot(bad_q, c="red")
        ax.plot(goal_q, c="green")
        for r in result:
            if r != len(agent_q):
                ax.axvline(x=r, color='k', linestyle="--")
        plt.show()
        return result

    # Explanation generation => (Phase 2)
    def diff_lists(self, l1, l2):
        """
        This is a function to substract l2 from l1.
                Args:
                        l1 (list of floats): A list of floats.
                        l2 (list of floats): A list of floats.

                Returns:
                        combined (list of floats): The result of the substraction as a list of floats in the size of the longest list.

        """
        l = max(len(l1), len(l2))
        combined = [0 for i in range(l)]
        for i in range(l):
            if i < len(l1):
                combined[i] += l1[i]
            if i < len(l2):
                combined[i] -= l2[i]
        return combined

    # Explanation generation => (Phase 2)
    def combine_segmentation(self, main_q, counter_q, main_bad_q, counter_bad_q, main_goal_q, counter_goal_q):
        """
        This is a function to calculate the segmentation of the combined lists of both the main and the counterfactual paths.
            Args:
                main_q (list of floats): list of q values calculated by agent for the main path
                counter_q (list of floats): list of q values calculated by agent for the counter path
                main_bad_q (list of floats): list of q values calculated by bad influence predictor for the main path
                counter_bad_q (list of floats): list of q values calculated by bad influence predictor for the counter path
                main_goal_q (list of floats): list of q values calculated by goal influence predictor for the main path
                counter_goal_q (list of floats): list of q values calculated by goal influence predictor for the counter path
             Returns:
                result (list of integers): List of all change points indices.
        """
        # Combination Method Diff:
        diff_q = self.diff_lists(main_q, counter_q)
        diff_bad_q = self.diff_lists(main_bad_q, counter_bad_q)
        diff_goal_q = self.diff_lists(main_goal_q, counter_goal_q)

        return self.change_point_detection(diff_q, diff_bad_q, diff_goal_q)

    # Influence predictor helper function
    # Use influence predictors to process simulated path
    def get_path_q_values(self, all_pos, all_dirs):
        """
        Get all the Q values and observation descriptions of all states in a trajectory.
        Args:
            all_pos: a list of tuples (x,y) representing agent's positions
            all_dirs: a list of integers representing agent's directions
        Returns:
            bad_q: List containing the negative influence values for every state.
            goal_q: List containing the positive influence values for every state.
            other_q: List containing the other influence values for every state. (Other is a tiny continuos )
            obs_desc: List containing the observation descriptions (each observation description is a list of tuples (object_type, color))
        """
        self.env.reset()
        bad_qs = []
        goal_qs = []
        # other_qs = []
        obs_desc = []
        string_print = "".join(["(" + str(i) + "," + str(j) + ")" for i, j in all_pos]) + "\n"
        for k in range(len(all_pos)):
            i = all_pos[k]
            di = all_dirs[k]
            if self.env.grid.get(*i) is None:
                obs, action, q, bad_q, goal_q = self.ipm.get_qs_of_obs(self.env, i, di, self.agent)
                string_print += str(self.env.agent_dir)
                bad_qs.append(bad_q)
                goal_qs.append(goal_q)
                # other_qs.append(other_q)
                features = describe_observation(obs['image'])
                obs_desc.append([t[1] + " " + t[0] for t in features])  # Get object and color "Blue ball"
        string_print += "\n" + " ".join(["{:.2f}".format(f) for f in bad_qs]) + "\n"
        string_print += " ".join(["{:.2f}".format(f) for f in goal_qs]) + "\n"
        print(string_print)
        return bad_qs, goal_qs, obs_desc

    def get_path_influence_actions(self, all_pos, all_dirs, threshold=-float("inf")):
        """
        Get all the predictions from the influence predictors of all states in a trajectory.
        Args:
            all_pos: a list of tuples (x,y) representing agent's positions
            all_dirs: a list of integers representing agent's directions
        Returns:
            max_actions: list of predicted actions from the influence predictors. list(int)
            """
        self.env.reset()
        agent_actions = []
        max_actions = []
        for k in range(len(all_pos)):
            i = all_pos[k]
            di = all_dirs[k]
            if self.env.grid.get(*i) is None:
                obs, max_action, action = self.ipm.get_action_of_obs(self.env, i, di, self.agent, threshold)
                agent_actions.append(action)
                max_actions.append(max_action)
        return max_actions, agent_actions

    def compare_influence_predictors_and_agent_actions_along_path(self, main_t, threshold=-float("inf")):
        main_agent_poss = self.agent_poss(main_t, self.obss_seqs)
        main_agent_dirs = self.agent_dirs(main_t, self.obss_seqs)
        max_actions, agent_actions = self.get_path_influence_actions(
            main_agent_poss, main_agent_dirs, threshold)
        return max_actions, agent_actions

    # Explanation generation => full_trajectory

    def get_q_changes_of_path(self, path_t):
        # Get all Qs
        obss, qs = self.q_changes(path_t, self.obss_seqs)
        agent_poss = self.agent_poss(path_t, self.obss_seqs)
        agent_dirs = self.agent_dirs(path_t, self.obss_seqs)
        bad_q, goal_q, obs_desc = self.get_path_q_values(
            agent_poss, agent_dirs)
        return qs, bad_q, goal_q, agent_poss, agent_dirs, obs_desc, obss

    def move_left(self):
        return [0, 2]

    def move_right(self):
        return [1, 2]

    def execute_action_seq(self, action_seq, len_explanation, obs_cur, path_name, obs=None):
        if not action_seq:
            raise IOError("Missing action sequence.")
        print("Action seq: ", action_seq)
        reward, done, info = None, None, None
        for action in action_seq:
            _, q_value = self.agent.get_action_and_values(obs, argmax=True)
            obs, reward, done, info = self.env.step(action)
            agent_pos = self.env.agent_pos
            agent_dir = self.env.agent_dir
            obs_cur = self.explanation_graph.add_observation(action, obs, reward, done, obs_cur, path_name, q_value,
                                                             agent_pos, agent_dir)
            len_explanation += 1
        i = len(action_seq) - 1
        return i, obs_cur, len_explanation, obs, reward, done, info

    def get_path(self, original_path_observations, j, len_main, direction="custom", action_seq=None):
        # Direction is a string and it can be either: other, random, right, left or forward.
        pre_obs, next_obs, previous_action, q_value, agent_pos, agent_dir = original_path_observations[j]
        path_name = "counterf_" + str(j) + "_" + direction
        self.explanation_graph.create_path(path_name, "explanation", "starting from " + str(j) + " Moving " + direction)
        #
        obs = self.env.reset()
        obs_cur = None
        # 1. Load all previous observations from the original path.
        pre_actions = []
        for k in range(j + 1):
            _, _, k_previous_action, _, _, _ = original_path_observations[k]
            if k == j:
                break
            pre_actions.append(k_previous_action.action)

        if pre_actions:
            _, obs_cur, _, obs, reward, done, info = self.execute_action_seq(pre_actions, 0, None,
                                                                             path_name, obs)

        if previous_action:  # if not last action?
            if direction == "random":
                random_action, q_value = self.agent.get_action_and_values(
                    {'image': self.explanation_graph.observations[
                        obs_cur.obs_id].obs_image,
                     'mission': self.env.mission, 'direction': self.env.agent_dir},
                    argmax=False)  # Allow only one random action.
                obs, reward, done, info = self.env.step(random_action)
                agent_pos = self.env.agent_pos
                agent_dir = self.env.agent_dir
                obs_cur = self.explanation_graph.add_observation(random_action, obs, reward, done, obs_cur, path_name,
                                                                 q_value,
                                                                 agent_pos, agent_dir)
                len_explanation = j + 1
                i = 0
            else:
                len_explanation = j
                if direction == "forward":
                    action_seq = [2]
                elif direction == "left":
                    action_seq = self.move_left()
                elif direction == "right":
                    action_seq = self.move_right()

                i, obs_cur, len_explanation, obs, reward, done, info = self.execute_action_seq(action_seq,
                                                                                               len_explanation,
                                                                                               obs_cur,
                                                                                               path_name, obs)

            while True:
                i += 1
                if done:
                    self.explanation_graph.paths[path_name].final_reward = reward
                    break
                if len_explanation >= len_main + EXTRA_LENGTH:
                    self.explanation_graph.paths[path_name].final_reward = reward
                    break
                # This replaces get_actions method from the agent.
                action, q_value = self.agent.get_action_and_values(obs)
                # ---
                obs, reward, done, _ = self.env.step(action)
                agent_pos = self.env.agent_pos
                agent_dir = self.env.agent_dir
                obs_cur = self.explanation_graph.add_observation(action, obs, reward, done, obs_cur, path_name, q_value,
                                                                 agent_pos,
                                                                 agent_dir)
                len_explanation += 1
        # Update sequences with new path
        self.obss_seqs[path_name] = {'path': self.explanation_graph.paths[path_name].get_path_array(),
                                     'reward': self.explanation_graph.paths[path_name].final_reward}
        return path_name

    def explanation_full_trajectory(self, main_t, counter_t):
        str_explanation = ""
        print("MAIN")
        main_qs, main_bad_q, main_goal_q, main_agent_poss, main_agent_dirs, main_obs_desc, main_obss = self.get_q_changes_of_path(
            main_t)
        counter_qs, counter_bad_q, counter_goal_q, counter_agent_poss, counter_agent_dirs, counter_obs_desc, counter_obss = self.get_q_changes_of_path(
            counter_t)
        # Main Q Segmentation:
        # change_point_detection(main_q, main_bad_q, main_goal_q)

        # Counter Q Segmentation:
        # change_point_detection(counter_q, counter_bad_q, counter_goal_q)

        # Diff Q Segmentation:

        # combine_segmentation(main_q, counter_q,
        #                     main_bad_q, counter_bad_q,
        #                     main_goal_q, counter_goal_q)

        # Visualize paths
        self.visualize_path(self.env, main_agent_poss, counter_agent_poss)

        # COST Based explanation: calculate scores
        main_reward = self.obss_seqs[main_t]['reward']
        counter_reward = self.obss_seqs[counter_t]['reward']
        main_average_q = self.get_average_q(main_qs)  # threshold
        counter_average_q = self.get_average_q(counter_qs)  # threshold
        q_diff, len_diff = self.measure_diff(main_qs,
                                             counter_qs)  # Measure difference between two trajectories. (reward, length)

        print("Main Reward: ", main_reward)
        print("Counter Reward: ", counter_reward)

        print("Main average Q: ", main_average_q)
        print("Counter average Q: ", counter_average_q)

        print("Q difference: ", q_diff)
        print("Length Difference: ", len_diff)

        if q_diff > 0:
            str_explanation += "The agent doesn't prefer this path as it generally gives less reward. "
        if len_diff < 0:
            str_explanation += "The agent prefers shorter paths. "

            # Avoiding "bad" objects explanation: compare Qs newObs Lava Q > originalObs Lava Q, originalObs Goal Q > newObs Goal Q
        main_average_bad_q = self.get_average_q(main_bad_q)  # threshold
        counter_average_bad_q = self.get_average_q(counter_bad_q)  # threshold
        bad_q_diff, _ = self.measure_diff(main_bad_q,
                                          counter_bad_q)  # Measure difference between two trajectories. (reward, length)

        main_average_good_q = self.get_average_q(main_goal_q)  # threshold
        counter_average_good_q = self.get_average_q(counter_goal_q)  # threshold
        good_q_diff, _ = self.measure_diff(main_goal_q,
                                           counter_goal_q)  # Measure difference between two trajectories. (reward, length)

        print("Main average bad Q: ", main_average_bad_q)
        print("Counter average bad Q: ", counter_average_bad_q)

        print("Main average goal Q: ", main_average_good_q)
        print("Counter average goal Q: ", counter_average_good_q)

        print("bad Q difference: ", bad_q_diff)
        print("goal Q difference: ", good_q_diff)

        if bad_q_diff < 0:
            str_explanation += "The counter path is more influenced by bad objects than the main path. "

        if good_q_diff > 0:
            str_explanation += "The counter path is less influenced by the goal than the main path. "
        print("----------------------")
        print(str_explanation)

        # if diff > THRESHOLD:
        # all_counter_q = get_all_q(counter_t)
        # return str_explanation(Ordered_obs)
        #    pass
        # else:
        # No interesting observations or proposed is better than current
        #    pass
