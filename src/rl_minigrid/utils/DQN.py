import torch
import torch.nn as nn
from collections import namedtuple
import numpy as np
import random


#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 1
### Data structure for holding experiences for replay
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'greedy_action'))


def prepare_state(state):
  state = state.transpose((2, 0, 1))
  state = torch.from_numpy(state).float() #TODO: Do we need to divide by 255?
  state = state.unsqueeze(0).to(DEVICE)
  return state

class DQN(nn.Module):

    ### Create all the nodes in the computation graph.
    ### We won't say how to put the nodes together into a computation graph. That is done
    ### automatically when forward() is called.
    def __init__(self, h, w, num_actions, num_reward_classes=1):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.num_reward_classes = num_reward_classes

        # This function returns the number of Linear input connections, which
        # depends on output of conv2d layers and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=2, stride=1): #TODO: Fix this size function.
            return (size - (kernel_size - 1) - 1) // stride + 1

        self.conv1 = nn.Conv2d(3, 16, kernel_size=2 , stride=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
        self.mp1 = nn.MaxPool2d(2, 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2 , stride=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2 , stride=1)
        nn.init.xavier_uniform_(self.conv3.weight, gain=np.sqrt(2))
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.relu3 = nn.LeakyReLU()

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32 #64, 52, 52
        linear_input_size = 64 * 52 * 52
        #print("DQN with size: ", linear_input_size)
        #assert linear_input_size == 173056 # Last thing I did here was I but back the size functions and put and assertion
        # CHECK: https://tianshou.readthedocs.io/en/v0.2.4/_modules/tianshou/utils/net/discrete.html
        #linear_input_size = 173056

        self.fc1 = nn.Linear(linear_input_size, 256)
        nn.init.xavier_uniform_(self.fc1.weight, gain=np.sqrt(2))
        self.relu4 = nn.LeakyReLU()
        # Create heads for actions for global reward and heads for lavaball influence
        self.head = nn.Linear(256, num_actions * num_reward_classes)
        nn.init.xavier_uniform_(self.head.weight, gain=np.sqrt(2))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        q_values = None
        x = self.mp1(self.relu1(self.conv1(x)))
        #print("Shape after first layer", x.shape)
        x = self.relu2(self.conv2(x))
        #print("Shape after second layer", x.shape)
        x = self.relu3(self.conv3(x))
        #print("Shape after third layer", x.shape)
        x = x.view(x.size(0), -1)
        #print("modified shape", x.shape)
        x = self.fc1(x)
        #print("fully connected shape", x.shape)
        x = self.relu4(x)
        q_values = self.head(x)
        return q_values  # Actions repeated for each reward class (num_actions * num_reward_classes)

CRITERION = nn.SmoothL1Loss()

policy_net_g = None
replay_memory_g = None
batch_g = None

### Take a DQN and do one forward-backward pass.
### Since this is Q-learning, we will run a forward pass to get Q-values for state-action pairs and then
### give the true value as the Q-values after the Q-update equation.
def optimize_model(policy_net, target_net, replay_memory, optimizer, batch_size, gamma, num_actions):
  global policy_net_g, batch_g
  if len(replay_memory) < batch_size:
    return

  # Sample from replay memory
  transitions = replay_memory.sample(batch_size)
  # Make a batch
  batch = Transition(*zip(*transitions))
  # Mask out batches that ended in done (next state is none)
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
  # Identify which batches are done
  non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
  # Make tensors for stuff in batch
  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)
  # Predict the q values for each action and each reward class
  # We are going to fold this so that the q values for the global reward problem are in one row and
  # the lavaball influence is in another row. Remove the .view if you don't have multiple reward classes.
  # And skip all the folding, unfolding, and duplicating (but do the gather).
  '''
  state_qs = policy_net(state_batch).view(-1, num_reward_classes, num_actions) # Fold the 6 q-values into 3x2
  state_qs_unfolded = torch.cat((state_qs[:,0,:], state_qs[:,1,:])) # only works for 2 reward classes
  #action_batch_duplicated = torch.cat((greedy_action_batch, greedy_action_batch)) # only works for 2 reward classes
  action_batch_duplicated = torch.cat((action_batch, action_batch)) # only works for 2 reward classes
  state_action_values_unfolded = state_qs_unfolded.gather(1, action_batch_duplicated)
  state_action_values = torch.cat((state_action_values_unfolded[0:batch_size], 
                                   state_action_values_unfolded[batch_size:]), dim=1) # refolding. Only works for 2 reward classes
  '''
  state_qs = policy_net(state_batch)
  state_action_values = state_qs.gather(1, action_batch.repeat([1, 1]) + (torch.tensor([list(range(0, 1))]).to(DEVICE)*num_actions).repeat([batch_size,1]))
  # Predict the q v alues for the successor state. Pack into a tensor according to mask
  next_state_values = torch.zeros([batch_size, 1], device=DEVICE)
  next_state_values[non_final_mask] = target_net(non_final_next_states).view(-1, 1, num_actions).max(dim=2)[0].detach() #fold the 6 q-values into 3x2
  # Bellmans
  expected_state_action_values = (next_state_values * gamma) + reward_batch
  # Compute loss
  policy_net_g = policy_net
  batch_g = batch
  loss = CRITERION(state_action_values, expected_state_action_values) # + CRITERION(state_action_values[:,0], state_action_values[:,1:].sum(dim=1))
  '''
  # The old code from q-learning without different reward classes.
  transitions = replay_memory.sample(batch_size)
  batch = Transition(*zip(*transitions))
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=DEVICE, dtype=torch.bool)
  non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)
  state_action_values = policy_net(state_batch).gather(1, action_batch)
  next_state_values = torch.zeros(batch_size, device=DEVICE)
  next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
  expected_state_action_values = (next_state_values * GAMMA) + reward_batch
  # Need to change above for decomposition
  # sum across the C dimension target_net(non_final_next_states) then max across what remains
  # Not sure though, loss should be num_actions x classes matrix. Can I do that? Might need to flatten.
  criterion = nn.SmoothL1Loss()
  loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
  '''
  optimizer.zero_grad()
  loss.backward()
  for param in policy_net.parameters():
      param.grad.data.clamp_(-1, 1)
  optimizer.step()
  return loss.item()


### Store transitions to use to prevent catastrophic forgetting.
### ReplayMemory implements a ring buffer. Items are placed into memory
###    until memory reaches capacity, and then new items start replacing old items
###    at the beginning of the array.
### Member variables:
###    capacity: (int) number of transitions that can be stored
###    memory: (array) holds transitions (state, action, next_state, reward)
###    position: (int) index of current location in memory to place the next transition.
class ReplayMemory(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  ### Store a transition in memory.
  def push(self, state, action, next_state, reward, greedy_action):
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = Transition(state, action, next_state, reward, greedy_action)
    self.position = (self.position + 1) % self.capacity

  ### Return a batch of transition objects from memory containing batch_size elements.
  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  ### This allows one to call len() on a ReplayMemory object. E.g. len(replay_memory)
  def __len__(self):
    return len(self.memory)