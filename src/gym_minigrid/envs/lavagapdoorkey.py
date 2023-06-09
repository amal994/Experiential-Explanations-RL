from src.gym_minigrid.minigrid import *
from src.gym_minigrid.register import register
import numpy as np

MAXIMUM_SIZE_F = 10 # Gap needs only 4, when door key involved make it 10.


class LavaGapDoorKeyEnv(MiniGridEnv):
    """
    Environment with a door and key with one wall of lava with a small gap to cross through
    sparse reward
    """
    def __init__(self, size, obstacle_type=Lava, seed=None):
        self.obstacle_type = obstacle_type
        self.fixed_env = False
        self.simple_reward = False
        self.no_door_key = False # True for gap.

        super().__init__(
            grid_size=size,
            max_steps=MAXIMUM_SIZE_F*size*size,
            # Set this to True for maximum speed
            see_through_walls=False,
            seed=seed
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        if self.fixed_env:
            # Create a vertical splitting wall
            splitIdx = width//2 - 1
            self.gap_pos = np.array((2,3))
            self.grid.vert_wall(splitIdx+3, 2, width // 3, self.obstacle_type)
            doorIdx = height//2 - 1

        else:
            # Create a vertical splitting wall
            splitIdx = self._rand_int(2, width-2)
            # Place the obstacle wall
            if splitIdx > width // 3 + 1:
                self.gap_pos = np.array((
                    self._rand_int(1, width // 3),
                    self._rand_int(1, height - 2),
                ))
                self.grid.horz_wall(self.gap_pos[0], self.gap_pos[1], width // 3, self.obstacle_type)
            else:
                self.gap_pos = np.array((
                    self._rand_int(1, width - 2),
                    self._rand_int(1, height // 3),
                ))
                self.grid.vert_wall(self.gap_pos[0], self.gap_pos[1], height // 3, self.obstacle_type)
            doorIdx = self._rand_int(1, height - 2)
            # Place a door in the wall
            while abs(doorIdx - self.gap_pos[1]) < 2:
                doorIdx = self._rand_int(1, height-2)


        self.grid.vert_wall(splitIdx, 0)
        if self.no_door_key:
            # Put a hole in the wall
            self.grid.set(splitIdx, doorIdx, None)
        else:
            self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)
            if self.fixed_env:
                self.put_obj(Key('yellow'), 2, 4)
            else:
                # Place a yellow key on the left side
                self.place_obj(
                    obj=Key('yellow'),
                    top=(0, 0),
                    size=(splitIdx, height)
                )

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        self.mission = "Avoid the lava and use the key to open the door and then get to the goal"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        agent_pos = self.agent_pos
        object = self.grid.get(agent_pos[0], agent_pos[1])
        if object.type == "lava":
            return -1 # Add Negative reward for stepping on Lava.

        if self.simple_reward:
            return 1
        else:
            return (1 - 0.9 * (self.step_count / self.max_steps)) * 10

class LavaGapDoorKeyEnv5x5(LavaGapDoorKeyEnv):
    def __init__(self):
        super().__init__(size=5)


class LavaGapDoorKeyEnv6x6(LavaGapDoorKeyEnv):
    def __init__(self):
        super().__init__(size=6)


class LavaGapDoorKeyEnv8x8(LavaGapDoorKeyEnv):
    def __init__(self):
        super().__init__(size=8)


class LavaGapDoorKeyEnv16x16(LavaGapDoorKeyEnv):
    def __init__(self):
        super().__init__(size=16)

register(
    id='MiniGrid-LavaGapDoorKeyEnv5x5-v0',
    entry_point='gym_minigrid.envs:LavaGapDoorKeyEnv5x5'
)

register(
    id='MiniGrid-LavaGapDoorKeyEnv6x6-v0',
    entry_point='gym_minigrid.envs:LavaGapDoorKeyEnv6x6'
)

register(
    id='MiniGrid-LavaGapDoorKeyEnv8x8-v0',
    entry_point='gym_minigrid.envs:LavaGapDoorKeyEnv8x8'
)

register(
    id='MiniGrid-LavaGapDoorKeyEnv16x16-v0',
    entry_point='gym_minigrid.envs:LavaGapDoorKeyEnv16x16'
)
