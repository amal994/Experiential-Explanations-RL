"""
Trajectory simulator helper functions

Description:
In this file we generate all the observations from our environments
to create the simulationsfor the different counterfactual trajectories
that can be generated.
"""
import os

import torch
import copy
import random
from tqdm import tqdm

import src.rl_minigrid.utils as utils
from src.explanation_generation.graph_viz import ExplanationGraph

import pickle

from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import save_img

import gym

EXTRA_LENGTH = 20
MAX_LENGTH = 25
SEED = 48


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def extract_dict_q_values_obs(obs_ids, paths):
    # Given a list of obs_ids and paths return the q_values of the outgoing actions of each of these obs.
    len_paths = len(paths)
    obs_q_dict = {i: {} for i in obs_ids}
    path_names = list(paths.keys())
    for p in range(len_paths):
        qs = paths[path_names[p]].get_q_values()
        for q in qs:
            try:
                obs_q_dict[q[0]][path_names[p]].append(q[1])
            except KeyError:
                obs_q_dict[q[0]][path_names[p]] = [q[1]]

    return obs_q_dict


def extract_q_values_obs(obs_ids, paths):
    # Given a list of obs_ids and paths return the q_values of the outgoing actions of each of these obs. 
    len_paths = len(paths)
    obs_q_dict = {i: [] for i in obs_ids}
    path_names = list(paths.keys())
    for p in range(len_paths):
        qs = paths[path_names[p]].get_edges_with_q_values()
        for q in qs:
            obs_q_dict[q[0]].append(q[3])
    return obs_q_dict


def get_action_and_values(agent, obs, random_action=None, argmax=True):
    # This is the get_actions method from the agent. 
    process_obs = agent.preprocess_obss([obs], device=agent.device)
    dist, value, agent.memories = agent.acmodel(process_obs, agent.memories)

    if random_action:
        action = random_action.value
    elif argmax:
        action = dist.probs.max(1, keepdim=True)[1][0].item()
    else:
        # When the model is well-trained, the probs of getting the correct action are 90% +.
        # Therefore, sample, would perform just like max almost all the time.
        # action = dist.sample()[0].item()
        # action = torch.topk(dist.probs, 2)[1][0][1].item()
        action = random.randint(0, dist.probs.shape[1] - 2)

    return action, value.item()


def move_agent_to_location(env, target_location):
    agent_pos = (env.agent_pos[0], env.agent_pos[1])
    horizontal_dir = None  # None, 'right', 'left'
    vertical_dir = None  # None, 'up', 'down'
    # Figure out whether to go left or right + Number of steps.
    horizontal_distance = target_location[0] - agent_pos[0]
    if horizontal_distance < 0:
        pass
    # Figure out whether to go up or down + Number of steps.
    vertical_distance = target_location[1] - agent_pos[1]
    # TODO: CHECK THIS

    pass


def get_path(explanation_g, env, original_path_observations, j, random_chain, len_main):
    pre_obs, next_obs, previous_action, q_value, agent_pos, agent_dir = original_path_observations[j]
    if random_chain > 0:
        path_name = "explore_" + str(j) + "_random_chain"
    else:
        path_name = "explore_" + str(j)
    explanation_g.create_path(path_name, "explanation", "starting from " + str(j))
    obss = [pre_obs]
    for k in range(j + 1):
        k_pre_obs, k_next_obs, k_previous_action, k_q_value, k_agent_pos, k_agent_dir = original_path_observations[k]
        if k == 0:
            obs_cur = explanation_g.paths[path_name].init_node(k_pre_obs, k_agent_pos, k_agent_dir)
            obs_cur.q_value = k_q_value
            obss.append(obs_cur.obs_id)
            if j == 0:
                break
            obs_cur.q_value = k_q_value
            obs_cur = explanation_g.paths[path_name].add_edge(obs_cur, k_next_obs, k_previous_action, k_q_value,
                                                              k_agent_pos, k_agent_dir)
        else:
            if k == j:
                break
            obs_cur.q_value = k_q_value
            obs_cur = explanation_g.paths[path_name].add_edge(obs_cur, k_next_obs, k_previous_action, k_q_value,
                                                              k_agent_pos, k_agent_dir)
            obss.append(obs_cur.obs_id)

    if previous_action:  # if not last action?
        random_action, q_value = get_action_and_values(agent,
                                                       {'image': explanation_g.observations[obs_cur.obs_id].obs_image,
                                                        'mission': env.mission, 'direction': env.agent_dir},
                                                       argmax=False)  # Allow only one random action.
        try:
            assert q_value == k_q_value
        except:
            print(q_value, "!=", obs_cur.q_value)
            raise AssertionError
        obs, reward, done, info = env.step(random_action)
        agent_pos = env.agent_pos
        agent_dir = env.agent_dir
        obs_cur = explanation_g.add_observation(random_action, obs, reward, done, obs_cur, path_name, q_value,
                                                agent_pos, agent_dir)
        obss.append(obs_cur.obs_id)
        len_explanation = j + 1
        i = 0
        while True:
            #    img = array_to_img(env.render(mode='rgb', tile_size=16))
            #    save_img("failures/seed_"+str(SEED)+"/"+path_name+"_" + str(i) + ".png", img)
            i += 1
            if done:
                explanation_g.paths[path_name].final_reward = reward
                break
            if len_explanation >= len_main + EXTRA_LENGTH:
                explanation_g.paths[path_name].final_reward = reward
                break
            # This replaces get_actions method from the agent.
            if random_chain > 0:
                action, q_value = get_action_and_values(agent, obs, argmax=False)  # Sample random action
                random_chain -= 1
            else:
                action, q_value = get_action_and_values(agent, obs, argmax=argmax)
            # ---
            obs, reward, done, _ = env.step(action)
            agent_pos = env.agent_pos
            agent_dir = env.agent_dir
            obs_cur = explanation_g.add_observation(action, obs, reward, done, obs_cur, path_name, q_value, agent_pos,
                                                    agent_dir)
            obss.append(obs_cur.obs_id)
            len_explanation += 1
    return explanation_g


def get_path_passing_point(explanation_g, env, original_path_observations, j, random_chain, len_main):
    pass


def run_simulations_with_one_env(env, agent, one_env=True, random_chain=0):
    explanation_graphs_in = []
    total_paths_in = dict()
    total_observations_in = dict()

    # Save initial environment states.
    if one_env:
        env.seed(SEED)
    init_obs = env.reset()
    init_env = copy.deepcopy(env)
    print("Agent location:", str(env.agent_pos))
    #  Save initial env setup.
    explanation_g = ExplanationGraph(init_env)
    img = array_to_img(init_env.render(mode='rgb', tile_size=16))
    save_img("main_env.png", img)
    # Create an agent path and let agent reach the goal
    path_name = "main"
    explanation_g.create_path(path_name, "main",
                              'Main agent path before any explanations.')  # Change to more detailed description.
    obs = init_obs
    env = copy.deepcopy(init_env)
    # Frozen envs:
    frozen_envs = [copy.deepcopy(init_env)]
    pre_obs = None
    agent_pos = env.agent_pos
    agent_dir = env.agent_dir
    pre_obs = explanation_g.add_observation(None, obs, None, None, pre_obs, path_name, None, agent_pos,
                                            agent_dir)

    # Main Path loop
    while True:
        # This replaces get_actions method from the agent. action = agent.get_action(obs)
        action, q_value = get_action_and_values(agent, obs, argmax=argmax)
        # ---
        obs, reward, done, info = env.step(action)
        agent_pos = env.agent_pos
        agent_dir = env.agent_dir
        frozen_envs.append(copy.deepcopy(env))
        # print("First main action:", action)
        pre_obs = explanation_g.add_observation(action, obs, reward, done, pre_obs, path_name, q_value, agent_pos,
                                                agent_dir)
        if done:
            explanation_g.paths[path_name].final_reward = reward
            # print(reward)
            break
    # Make a copy of the main path observations.
    original_path_observations = explanation_g.paths[path_name].get_edges_with_q_values()
    len_main = len(original_path_observations)
    # 1. Generate All possible explanations: From each observation generate a trajectory.
    # print("Main Path ", str(i), ' created. len: ', len_main)
    if len_main <= MAX_LENGTH:
        for j in tqdm(range(len_main)):
            env = frozen_envs[j]
            # At each step, create two counterfactual paths.
            # One with only one random action
            explanation_g = get_path(explanation_g, copy.deepcopy(env), original_path_observations, j, 0, len_main)
            # And the other with multiple random actions.
            explanation_g = get_path(explanation_g, copy.deepcopy(env), original_path_observations, j, random_chain,
                                     len_main)
            # And another using unvisited_obs. 
            # 1. Check if there are unvisited nearby observations O. 
            # 2. Force agent to visit O before proceeding to the goal. 
            # 3. Collect new_paths and observations and updated visited/unvisited stacks. 
            # explanation_g = get_path_passing_point(explanation_g,  copy.deepcopy(env), original_path_observations, j, random_chain, len_main)
        # End episode by recording paths and observations:
        total_observations_in.update(explanation_g.observations)
        total_paths_in.update(explanation_g.paths)
        explanation_graphs_in.append(explanation_g)
    # elif len_main > 800:
    #    img = array_to_img(env.render(mode='rgb', tile_size=16))
    #    save_img("failures/env_" + str(SEED) + "_" + str(len_main) + ".png", img)

    return explanation_graphs_in, total_paths_in, total_observations_in


def verify_paths_have_similar_obss(p_arrays, with_episodes=False):
    b = {}
    obs_dict = []
    for p in p_arrays:
        b[p] = []
        path_name_deconstruction = p.split("_")
        is_main = False
        if with_episodes:
            if path_name_deconstruction[1] == "main":
                label = path_name_deconstruction[1]
                is_main = True
            else:
                label = int(path_name_deconstruction[2])
        else:
            if path_name_deconstruction[0] == "main":
                label = path_name_deconstruction[0]
                is_main = True
            else:
                label = int(path_name_deconstruction[1])
        for t in p_arrays[p]['path']:
            item = t[0]
            b[p].append(item)
        if is_main:
            obs_dict = list(set(b[p]))
        temp = []
        for k in b[p]:
            try:
                i = obs_dict.index(k)
            except ValueError:
                obs_dict.append(k)
                i = obs_dict.index(k)
            if not is_main and len(temp) - 1 == label:
                temp.append("*" + str(i) + "*")
            else:
                temp.append(i)
        b[p] = temp
    for k in b:
        print(k, ":", str(b[k]))


def verify_paths_have_similar_actions(p_arrays, with_episodes=False):
    a = {}
    for p in p_arrays:
        a[p] = []
        for t in p_arrays[p]['path']:
            item = t[3]
            try:
                path_name_deconstruction = p.split("_")

                if with_episodes:
                    if path_name_deconstruction[1] == "main":
                        label = path_name_deconstruction[1]
                    else:
                        label = int(path_name_deconstruction[2])
                else:
                    if path_name_deconstruction[0] == "main":
                        label = path_name_deconstruction[0]
                    else:
                        label = int(path_name_deconstruction[1])

                if len(a[p]) == label:
                    a[p].append("*" + str(item) + "*")
                else:
                    a[p].append(item)
            except Exception:
                a[p].append(item)
    for k in a:
        print(k, ":", str(a[k]))


if __name__ == "__main__":
    # -------------------------------
    # Params
    # TODO: Make these into argument params
    env_name = "MiniGrid-LivingRoomEnv9x9-v0"
    env_seed = SEED
    bad = "blue"
    current_directory = utils.get_storage_dir()
    model_dir = os.path.join(current_directory, 'LivingRoomEnv9-' + bad)
    print(model_dir)
    memory = False  # Assume agent doesn't use LSTM
    text = False
    argmax = True  # This can get us the probabilities.
    episodes = 1
    env_fname = "LivingRoomEnv9"
    folders = [(env_fname + "-" + bad + "-demo", 0),
               (env_fname + "-" + bad + "-scene1", 1),
               (env_fname + "-" + bad + "-scene2", 2),
               (env_fname + "-" + bad + "-scene3", 3),
               (env_fname + "-" + bad + "-scene4", 4),
               (env_fname + "-" + bad + "-scene5", 5),
               (env_fname + "-" + bad + "-scene1_s", 6),
               (env_fname + "-" + bad + "-scene2_s", 7),
               (env_fname + "-" + bad + "-scene3_s", 8),
               (env_fname + "-" + bad + "-scene4_s", 9),
               (env_fname + "-" + bad + "-scene5_s", 10),
               ]

    for folder, scene in folders:
        print(folder)
        # -------------------------------
        # Define environment and agent
        env = gym.make(env_name, scene=scene, bad=bad)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        agent = utils.Agent(env.observation_space, env.action_space, model_dir, device=device, argmax=argmax,
                            use_memory=memory,
                            use_text=text)
        # img = array_to_img(env.render(mode='rgb', tile_size=16))
        # save_img("failures/env_new_lava.png", img)
        # exit(0)
        # -------------------------------
        # Define observation collection:
        fixed_list_q_values = dict()
        list_q_values = dict()

        # -------------------------------
        # Run scripts and collect observations and sequences
        explanation_graphs, total_paths, total_observations = run_simulations_with_one_env(env, agent, random_chain=10)
        print("Got scripts")
        # -------------------------------
        # Get Observations:

        obs_keys = list(total_observations.keys())
        print("Total number of observations", len(obs_keys))
        print("Total number of paths", len(total_paths))

        list_q_values.update(extract_dict_q_values_obs(obs_keys, total_paths))

        # -------------------------------
        # Get Paths:
        paths_arrays = dict()
        for p in total_paths:
            paths_arrays[p] = {'path': total_paths[p].get_path_array(), 'reward': total_paths[p].final_reward}

        # -------------------------------
        # SAVE OBJECTS
        folder_dir = os.path.join(current_directory, r'observations/' + folder)
        utils.create_folders_if_necessary(folder_dir)
        all_obs_q_directory = os.path.join(folder_dir, r'all_obs_q')
        utils.create_folders_if_necessary(all_obs_q_directory)
        save_obj(list_q_values, all_obs_q_directory)
        all_obss_directory = os.path.join(folder_dir, r'all_obss')
        utils.create_folders_if_necessary(all_obss_directory)
        save_obj(total_observations, all_obss_directory)
        sequential_obs_directory = os.path.join(folder_dir, r'sequential_obs')
        utils.create_folders_if_necessary(sequential_obs_directory)
        save_obj(paths_arrays, sequential_obs_directory)
        explanation_graph_directory = os.path.join(folder_dir, r'explanation_graph')
        utils.create_folders_if_necessary(explanation_graph_directory)
        save_obj(explanation_graphs[0], explanation_graph_directory)