import argparse
import os
import time
import datetime
import torch
import torch_ac
import torch.optim as optim

import tensorboardX
import sys

from src.gym_minigrid.specialRewardWrapper import SpecialRewardsWrapper

from src.rl_minigrid.utils.DQN import DQN, ReplayMemory, prepare_state, optimize_model
import src.rl_minigrid.utils as utils
from model import ACModel
import gym

if __name__ == "__main__":
    # Parse arguments

    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--algo", default='ppo',  # Amal removed: required=True,
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--env", default='MiniGrid-LivingRoomEnv9x9-v0',  # Amal removed: required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--bad_obj", default='blue',  # Amal added,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 1)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=22,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=1500000,
                        help="number of frames of training (default: 1e7)")

    ## Parameters for main algorithm
    parser.add_argument("--epochs", type=int, default=10,  # AMAL: CHANGED TO 10
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.99,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=3.5e-4,  # AMAL: Changed to 3.5e-4
                        help="learning rate (default: 0.001)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="add a GRU to the model to handle text input")
    # Additions to original Torch-ac code:
    parser.add_argument("--cache_obs", default=True,
                        help="Cache observations for the explanations")
    parser.add_argument("--optimize", default=True,
                        help="Option to optimize the hyperparams")
    parser.add_argument("--trials", default=True,
                        help="Number of trials for hyperparams optimization")
    # -------
    args = parser.parse_args()

    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    # default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"
    # default_model_name = "small_neg_" + default_model_name # Amal: Adding in the small negative rewards.
    # model_name = args.model or default_model_name
    bad = args.bad_obj
    model_name = "LivingRoomEnv9-" + bad
    model_dir = utils.get_model_dir(model_name)
    print("Model directory: ", model_dir)
    # Load loggers and Tensorboard writer
    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")

    # Save observations:
    observations = dict()
    seed = 48
    scenes = 11
    # Load environments
    envs = []
    for i in range(args.procs):
        scene = i % scenes
        print(scene)
        try:
            env = gym.make(args.env, scene=scene, bad=bad)
            env = SpecialRewardsWrapper(env)
        except AssertionError as e:
            print("Error at scene", scene)
            print(e)
            exit(0)
        envs.append(env)
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    # print("Observation Space")
    # print(envs[0].observation_space)
    # exit(0)
    # -----

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])

    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss)

    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    screen_width = obs_space["image"][0]
    screen_height = obs_space["image"][1]
    print("Screen width and height are", screen_width, " - ", screen_height)
    num_actions = envs[0].action_space.n
    num_reward_classes = 1
    replay_capacity = 50000  # How big is the replay memory
    gamma = 0.5  # How much to discount the future [0..1]
    target_update = 5  # Delays updating the network for loss calculations. 0=don't delay, or 1+ number of episodes
    BOOTSTRAP = 100000  # How many steps to run to fill up replay memory before training starts
    batch_size = 64  # How many replay experiences to run through neural net at once

    # Begin <
    # Influence Learners Initialization
    # ----------
    device_ids = list(range(torch.cuda.device_count()))
    dqn_device = 1
    # bad DQN

    bad_policy_net = DQN(screen_height, screen_width, num_actions, num_reward_classes).to(dqn_device)
    if "model_state" in status:
        bp_state_dict = torch.load(os.path.join(model_dir, "bad_dqn.model"))
        bad_policy_net.load_state_dict(bp_state_dict)
    # Target network is a snapshot of the policy network that lags behind (for stablity)
    bad_target_net = DQN(screen_height, screen_width, num_actions).to(dqn_device)

    bad_target_net.load_state_dict(bad_policy_net.state_dict())
    bad_target_net.eval()

    # Evaluation net
    bad_eval_net = DQN(screen_height, screen_width, num_actions).to(dqn_device)
    bad_eval_net.eval()

    # Instantiate the replay memory
    bad_replay_memory = ReplayMemory(replay_capacity)

    # bad optimizer
    bad_optimizer = optim.RMSprop(bad_policy_net.parameters(), lr=1e-5)

    bad_steps_done = 0  # How many steps have been run
    bad_best_eval = -float('inf')  # The best model evaluation to date
    bad_history = [(0.0, 0.0)]  # Keep track of eval performance

    # ----------

    # Goal DQN
    goal_policy_net = DQN(screen_height, screen_width, num_actions, num_reward_classes).to(dqn_device)
    # goal_policy_net = DDP(goal_policy_net, device_ids= device_ids)
    if "model_state" in status:
        gp_state_dict = torch.load(os.path.join(model_dir, "goal_dqn.model"))
        goal_policy_net.load_state_dict(gp_state_dict)
    # Target network is a snapshot of the policy network that lags behind (for stablity)
    goal_target_net = DQN(screen_height, screen_width, num_actions).to(dqn_device)
    # goal_target_net = DDP(goal_target_net, device_ids= device_ids)

    goal_target_net.load_state_dict(goal_policy_net.state_dict())
    goal_target_net.eval()
    # Evaluation net
    goal_eval_net = DQN(screen_height, screen_width, num_actions).to(dqn_device)
    # goal_eval_net = DDP(goal_eval_net, device_ids= device_ids)

    goal_eval_net.eval()

    # Instantiate the replay memory
    goal_replay_memory = ReplayMemory(replay_capacity)

    # Goal optimizer
    goal_optimizer = optim.RMSprop(goal_policy_net.parameters(), lr=1e-5)

    goal_steps_done = 0  # How many steps have been run
    goal_best_eval = -float('inf')  # The best model evaluation to date
    goal_history = [(0.0, 0.0)]  # Keep track of eval performance

    # ------
    # ##### no longer used!  #####
    # Other DQN
    # other_policy_net = DQN(screen_height, screen_width, num_actions, num_reward_classes).to(dqn_device)
    # other_policy_net = DDP(other_policy_net, device_ids= device_ids)

    # Target network is a snapshot of the policy network that lags behind (for stablity)
    # other_target_net = DQN(screen_height, screen_width, num_actions).to(dqn_device)
    # other_target_net = DDP(other_target_net, device_ids= device_ids)

    # other_target_net.load_state_dict(other_policy_net.state_dict())
    # other_target_net.eval()
    # Evaluation net
    # other_eval_net = DQN(screen_height, screen_width, num_actions).to(dqn_device)
    # other_eval_net = DDP(other_eval_net, device_ids= device_ids)

    # other_eval_net.eval()

    # Instantiate the replay memory
    # other_replay_memory = ReplayMemory(replay_capacity)

    # Other optimizer
    # other_optimizer = optim.RMSprop(other_policy_net.parameters(), lr=1e-5)

    # other_steps_done = 0               # How many steps have been run
    # other_best_eval = -float('inf')     # The best model evaluation to date
    # other_history = [(0.0, 0.0)]      # Keep track of eval performance

    # ------
    # End >
    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        bad_avg_loss = 0.0
        goal_avg_loss = 0.0
        # other_avg_loss = 0.0

        # HERE We can update the DQNs with the observations.
        # Exps has observations, actions, goal_rewards
        # states = [prepare_state(obs) for obs in exps.obss]
        # actions = [action for action in exps.obss]
        # rewards = [reward for reward in exps.obss]

        # If observation contain bad update bad DQN
        # bad_rewards = [reward if reward < 0 else 0 for reward in rewards]
        # bad_rewards = torch.tensor(bad_rewards, device=device).unsqueeze(dim=0)
        # Store the transition in memory
        for i in range(len(exps.reward)):
            action = exps.action[i]
            action = torch.tensor([action], device=dqn_device).unsqueeze(dim=0)
            reward = exps.reward[i]

            rendered_obs = env.get_obs_render(exps.obs.image[i].cpu().detach().numpy(),
                                              tile_size=8)
            state = prepare_state(rendered_obs)

            if i < len(exps.reward) - 1:
                rendered_obs = env.get_obs_render(exps.obs.image[i + 1].cpu().detach().numpy(),
                                                  tile_size=8)
                next_state = prepare_state(rendered_obs)
            else:
                next_state = None

            if reward == -1:  # If reward is negative: i.e. agent fell in bad
                bad_reward = -1 * reward  # Make bad Reward positive.
                goal_reward = 0
                # other_reward = 0
            elif reward >= 0:
                bad_reward = 0
                goal_reward = reward
                # other_reward = 0
            else:
                #    other_reward =  reward # add -1 * to Make small diverse Reward positive.
                bad_reward = 0
                goal_reward = 0

            bad_reward = torch.tensor([bad_reward], device=dqn_device).unsqueeze(dim=0)
            goal_reward = torch.tensor([goal_reward], device=dqn_device).unsqueeze(dim=0)
            # other_reward = torch.tensor([other_reward], device=dqn_device).unsqueeze(dim=0)

            bad_replay_memory.push(state, action, next_state, bad_reward, None)
            goal_replay_memory.push(state, action, next_state, goal_reward, None)
            # other_replay_memory.push(state, action, next_state, other_reward, None)

        for i in range(len(exps.reward) // batch_size):
            # If we are past bootstrapping we should perform one step of the optimization
            bad_avg_loss = bad_avg_loss + optimize_model(bad_policy_net,
                                                         bad_target_net if target_update > 0 else bad_policy_net,
                                                         bad_replay_memory, bad_optimizer, batch_size, gamma,
                                                         num_actions=num_actions)
            goal_avg_loss = goal_avg_loss + optimize_model(goal_policy_net,
                                                           goal_target_net if target_update > 0 else goal_policy_net,
                                                           goal_replay_memory, goal_optimizer, batch_size, gamma,
                                                           num_actions=num_actions)
            # other_avg_loss = other_avg_loss + optimize_model(other_policy_net,
            #                                               other_target_net if target_update > 0 else other_policy_net,
            #                                               other_replay_memory, other_optimizer, batch_size, gamma,
            #                                               num_actions=num_actions)

            bad_steps_done += 1
            goal_steps_done += 1
            # other_steps_done += 1
            # Should we update the target network?
            if target_update > 0 and bad_steps_done % target_update == 0:
                bad_target_net.load_state_dict(bad_policy_net.state_dict())
                goal_target_net.load_state_dict(goal_policy_net.state_dict())
                # other_target_net.load_state_dict(other_policy_net.state_dict())

        txt_logger.info("Average bad policy loss: " + str(bad_avg_loss))
        txt_logger.info("Average goal policy loss: " + str(goal_avg_loss))
        # txt_logger.info("Average other policy loss: " + str(other_avg_loss))

        # bad_replay_memory
        # If observation contain Key update Key DQN
        # If observation contain Door update Door DQN
        # If observation contain Goal update Goal DQN
        # goal_rewards = [reward if reward > 0 else 0 for reward in rewards]
        # goal_rewards = torch.tensor(goal_rewards, device=device).unsqueeze(dim=0)

        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs
        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
            print("Saving DQN...")
            torch.save(bad_policy_net.state_dict(), os.path.join(model_dir, "bad_dqn.model"))
            torch.save(goal_policy_net.state_dict(), os.path.join(model_dir, "goal_dqn.model"))
            # torch.save(other_policy_net, os.path.join(model_dir, "other_dqn.model"))
            print("Done saving.")
