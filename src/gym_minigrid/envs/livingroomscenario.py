"""

"""
from src.gym_minigrid.minigrid import *
from src.gym_minigrid.register import register
from src.gym_minigrid.wrappers import *

MAXIMUM_SIZE_F = 10 # Gap needs only 4, when door key involved make it 10.

class Circle(Ball):
    def __init__(self, color='purple'):
        super(Circle, self).__init__(color)

    def can_pickup(self):
        return False

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Square(Ball):
    def __init__(self, color='blue'):
        super(Square, self).__init__(color)

    def can_pickup(self):
        return False

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), COLORS[self.color])

class Triangle(Ball):
    def __init__(self, color='yellow'):
        super(Triangle, self).__init__(color)

    def can_pickup(self):
        return False

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_triangle((0.5,0.1), 
                                                             (0.1,0.9), 
                                                             (0.9,0.9)), COLORS[self.color])


class LivingRoomEnv(MiniGridEnv):

    def __init__(self, scene=0, size=7, bad='purple', r_agent = False):
        self.agent_start_dir = 0
        self.bad = bad
        self.scene = scene
        self.random_agent = r_agent

        super().__init__(
            grid_size=size,
            max_steps=10 * size * size,
            see_through_walls=False
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.mission = "get to the green goal square while avoiding the " + self.bad + " objects"
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.scene == 0:
            self.generate_demo()
        elif self.scene == 1:
            self.generate_scenario1()
        elif self.scene == 2:
            self.generate_scenario2()
        elif self.scene == 3:
            self.generate_scenario3()
        elif self.scene == 4:
            self.generate_scenario4()
        elif self.scene == 5:
            self.generate_scenario5()
        elif self.scene == 6:
            self.generate_scenario1_s()
        elif self.scene == 7:
            self.generate_scenario2_s()
        elif self.scene == 8:
            self.generate_scenario3_s()
        elif self.scene == 9:
            self.generate_scenario4_s()
        elif self.scene == 10:
            self.generate_scenario5_s()

    def generate_demo(self):
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, self.height - 2)

        # Build wall
        self.grid.horz_wall(2, self.height // 2, length=3)

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


    def generate_scenario1(self):
        # Place a goal square across the wall from agent
        self.put_obj(Goal(), self.width - 2, self.height // 2)

        # Build wall
        self.grid.vert_wall(2, 3, length=3)

        # Circle
        self.grid.set(self.width // 2 - 1, 1, Circle())

        # Triangle
        self.grid.set(self.width // 2, 1, Triangle())

        # Box
        self.grid.set(self.width // 2 - 1, self.height - 2, Square())

        # Triangle
        self.grid.set(self.width // 2, self.height - 2, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
    def generate_scenario1_s(self):
        # Place a goal square across the wall from agent
        self.put_obj(Goal(), self.width - 2, self.height // 2)

        # Build wall
        self.grid.vert_wall(2, 3, length=3)
        if self.bad == "blue":   # hide negative reward object
            # Circle
            self.grid.set(self.width // 2 - 1, 1, Circle())

        # Triangle
        self.grid.set(self.width // 2, 1, Triangle())

        if self.bad == "purple":   # hide negative reward object
            # Box
            self.grid.set(self.width // 2 - 1, self.height - 2, Square())

        # Triangle
        self.grid.set(self.width // 2, self.height - 2, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def generate_scenario2(self):
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, self.height // 2)

        # Build wall
        self.grid.vert_wall(2, 3, length=3)

        # Circle
        self.grid.set(1, 1, Circle())

        # Triangle
        self.grid.set(3, self.height // 2, Triangle())
        # Triangle
        self.grid.set(4, self.height // 2, Triangle())

        # Box
        self.grid.set(1, self.height - 2, Square())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
    def generate_scenario2_s(self):
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, self.height // 2)

        # Build wall
        self.grid.vert_wall(2, 3, length=3)

        if self.bad == "blue":   # hide negative reward object
            # Circle
            self.grid.set(1, 1, Circle())

        # Triangle
        self.grid.set(3, self.height // 2, Triangle())
        # Triangle
        self.grid.set(4, self.height // 2, Triangle())

        if self.bad == "purple":  # hide negative reward object
            # Box
            self.grid.set(1, self.height - 2, Square())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def generate_scenario3(self):
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, self.height // 2)

        # Build wall
        self.grid.horz_wall(2, self.height // 2, length=3)

        # Circle
        self.grid.set(3, 1, Circle())
        # Circle
        self.grid.set(4, 1, Circle())

        # Triangle
        self.grid.set(2, 1, Triangle())
        # Triangle
        self.grid.set(5, 1, Triangle())

        # Box
        self.grid.set(3, 7, Square())
        # Box
        self.grid.set(4, 7, Square())

        # Triangle
        self.grid.set(2, 7, Triangle())
        # Triangle
        self.grid.set(5, 7, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
    def generate_scenario3_s(self):
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, self.height // 2)

        # Build wall
        self.grid.horz_wall(2, self.height // 2, length=3)

        if self.bad == "blue":  # hide negative reward object
            # Circle
            self.grid.set(3, 1, Circle())
            # Circle
            self.grid.set(4, 1, Circle())

        # Triangle
        self.grid.set(2, 1, Triangle())
        # Triangle
        self.grid.set(5, 1, Triangle())

        if self.bad == "purple":  # hide negative reward object
            # Box
            self.grid.set(3, 7, Square())
            # Box
            self.grid.set(4, 7, Square())

        # Triangle
        self.grid.set(2, 7, Triangle())
        # Triangle
        self.grid.set(5, 7, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def generate_scenario4(self):
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, self.height - 2)

        # Build wall
        self.grid.vert_wall(3, 2, length=3)
        self.grid.vert_wall(1, 2, length=1)

        if self.bad == "purple":  # Force agent to go to the bad side
            # Circle
            self.grid.set(3, 6, Circle())
            # Box
            self.grid.set(5, 2, Square())
        else:
            # Circle
            self.grid.set(5, 2, Circle())
            # Box
            self.grid.set(3, 6, Square())

        # Triangle
        self.grid.set(self.width - 2, 1, Triangle())
        # Triangle
        self.grid.set(6, 1, Triangle())

        # Triangle
        self.grid.set(1, self.height - 2, Triangle())
        # Triangle
        self.grid.set(2, self.height - 2, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1,1)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
    def generate_scenario4_s(self):
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, self.height - 2)

        # Build wall
        self.grid.vert_wall(3, 2, length=3)
        self.grid.vert_wall(1, 2, length=1)

        if self.bad == "purple":  # Force agent to go to the bad side
            # Circle
            #self.grid.set(3, 6, Circle())
            # Box
            self.grid.set(5, 2, Square())
        else:
            # Circle
            self.grid.set(5, 2, Circle())
            # Box
            #self.grid.set(3, 6, Square())

        # Triangle
        self.grid.set(self.width - 2, 1, Triangle())
        # Triangle
        self.grid.set(6, 1, Triangle())

        # Triangle
        self.grid.set(1, self.height - 2, Triangle())
        # Triangle
        self.grid.set(2, self.height - 2, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1,1)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def generate_scenario4_a(self): # Add a wall to force agent to go to the bad side
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, self.height - 2)

        # Build wall
        self.grid.vert_wall(3, 2, length=3)
        self.grid.vert_wall(1, 2, length=1)

        self.grid.vert_wall(3, 1, length=1) # Block the way near the bad object
        if self.bad == "purple": # Force agent to go to the bad side
            # Circle
            self.grid.set(3, 6, Circle())
            # Box
            self.grid.set(5, 2, Square())
        else:
            # Circle
            self.grid.set(5, 2, Circle())
            # Box
            self.grid.set(3, 6, Square())

        # Triangle
        self.grid.set(self.width - 2, 1, Triangle())
        # Triangle
        self.grid.set(6, 1, Triangle())

        # Triangle
        self.grid.set(1, self.height - 2, Triangle())
        # Triangle
        self.grid.set(2, self.height - 2, Triangle())
        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1,1)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


    def generate_scenario4_b(self): # Remove wall from (3,2)
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, self.height - 2)

        # Build wall
        self.grid.vert_wall(1, 2, length=1)

        if self.bad == "purple":  # Force agent to go to the bad side
            # Circle
            self.grid.set(3, 6, Circle())
            # Box
            self.grid.set(5, 2, Square())
        else:
            # Circle
            self.grid.set(5, 2, Circle())
            # Box
            self.grid.set(3, 6, Square())

        # Triangle
        self.grid.set(self.width - 2, 1, Triangle())
        # Triangle
        self.grid.set(6, 1, Triangle())

        # Triangle
        self.grid.set(1, self.height - 2, Triangle())
        # Triangle
        self.grid.set(2, self.height - 2, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1,1)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def generate_scenario4_c(self): # Move Triangle from (1,6) to (3,5)
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, self.height - 2)

        # Build wall
        self.grid.vert_wall(3, 2, length=3)
        self.grid.vert_wall(1, 2, length=1)

        if self.bad == "purple":  # Force agent to go to the bad side
            # Circle
            self.grid.set(3, 6, Circle())
            # Box
            self.grid.set(5, 2, Square())
        else:
            # Circle
            self.grid.set(5, 2, Circle())
            # Box
            self.grid.set(3, 6, Square())

        # Triangle
        self.grid.set(self.width - 2, 1, Triangle())
        # Triangle
        self.grid.set(6, 1, Triangle())

        # Triangle
        self.grid.set(2, 6, Triangle())
        # Triangle
        self.grid.set(2, self.height - 2, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1,1)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


    def generate_scenario4_d(self): # Move triangle from (7,2) to (6,2)
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width - 2, self.height - 2)


        # Build wall
        self.grid.vert_wall(3, 2, length=3)
        self.grid.vert_wall(1, 2, length=1)

        if self.bad == "purple":  # Force agent to go to the bad side
            # Circle
            self.grid.set(3, 6, Circle())
            # Box
            self.grid.set(5, 2, Square())
        else:
            # Circle
            self.grid.set(5, 2, Circle())
            # Box
            self.grid.set(3, 6, Square())

        # Triangle
        self.grid.set(5, 3, Triangle())
        # Triangle
        self.grid.set(6, 1, Triangle())

        # Triangle
        self.grid.set(1, self.height - 2, Triangle())
        # Triangle
        self.grid.set(2, self.height - 2, Triangle())

        # Box
        self.grid.set(5, 2, Square())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1,1)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def generate_scenario5(self):
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width-2, self.height//2)


        # Build wall
        self.grid.horz_wall(3, self.height//2, length=3)

        if self.bad == "purple":  # Force agent to go to the bad side
            # Circle
            self.grid.set(5, 1, Circle())
            # Box
            self.grid.set(5, self.height - 2, Square())
        else:
            # Box
            self.grid.set(5, 1, Square())
            # Circle
            self.grid.set(5, self.height - 2, Circle())

        # Triangle
        self.grid.set(3, 1, Triangle())

        # Triangle
        self.grid.set(3, self.height - 2, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
    def generate_scenario5_s(self):
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width-2, self.height//2)


        # Build wall
        self.grid.horz_wall(3, self.height//2, length=3)

        if self.bad == "purple":  # Force agent to go to the bad side
            # Circle
            #self.grid.set(5, 1, Circle())
            # Box
            self.grid.set(5, self.height - 2, Square())
        else:
            # Box
            #self.grid.set(5, 1, Square())
            # Circle
            self.grid.set(5, self.height - 2, Circle())

        # Triangle
        self.grid.set(3, 1, Triangle())

        # Triangle
        self.grid.set(3, self.height - 2, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def generate_scenario5_a(self): # Add a wall in front of dangerous object
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width-2, self.height//2)

        # Build wall
        self.grid.horz_wall(3, self.height//2, length=3)

        if self.bad == "purple":  # Force agent to go to the bad side
            # Circle
            self.grid.set(5, 1, Circle())
            # Box
            self.grid.set(5, self.height - 2, Square())
        else:
            # Box
            self.grid.set(5, 1, Square())
            # Circle
            self.grid.set(5, self.height - 2, Circle())

        # Barrier
        self.grid.horz_wall(5, 2, length=1)

        # Triangle
        self.grid.set(3, 1, Triangle())

        # Triangle
        self.grid.set(3, self.height - 2, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def generate_scenario5_b(self): # Place a wall in front of neutral object
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width-2, self.height//2)


        # Build wall
        self.grid.horz_wall(3, self.height//2, length=3)

        if self.bad == "purple":  # Force agent to go to the bad side
            # Circle
            self.grid.set(5, 1, Circle())
            # Box
            self.grid.set(5, self.height - 2, Square())
        else:
            # Box
            self.grid.set(5, 1, Square())
            # Circle
            self.grid.set(5, self.height - 2, Circle())

        # Barrier
        self.grid.horz_wall(5, 6, length=1)

        # Triangle
        self.grid.set(3, 1, Triangle())

        # Triangle
        self.grid.set(3, self.height - 2, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


    def generate_scenario5_c(self): # barrier in front of triangle
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width-2, self.height//2)


        # Build wall
        self.grid.horz_wall(3, self.height//2, length=3)

        if self.bad == "purple":  # Force agent to go to the bad side
            # Circle
            self.grid.set(5, 1, Circle())
            # Box
            self.grid.set(5, self.height - 2, Square())
        else:
            # Box
            self.grid.set(5, 1, Square())
            # Circle
            self.grid.set(5, self.height - 2, Circle())

        # Barrier
        self.grid.horz_wall(3, 2, length=1)

        # Triangle
        self.grid.set(3, 1, Triangle())

        # Triangle
        self.grid.set(3, self.height - 2, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


    def generate_scenario5_d(self): # Move triangle from (7,2) to (6,2)
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), self.width-2, self.height//2)


        # Build wall
        self.grid.horz_wall(3, self.height//2, length=3)

        if self.bad == "purple":  # Force agent to go to the bad side
            # Circle
            self.grid.set(5, 1, Circle())
            # Box
            self.grid.set(5, self.height - 2, Square())
        else:
            # Box
            self.grid.set(5, 1, Square())
            # Circle
            self.grid.set(5, self.height - 2, Circle())

        # Barrier
        self.grid.horz_wall(3, 6, length=1)

        # Triangle
        self.grid.set(3, 1, Triangle())

        # Triangle
        self.grid.set(3, self.height - 2, Triangle())

        # Place the agent
        if not self.random_agent:
            self.agent_start_pos = (1, self.height // 2)
            # Place the agent
            self.agent_pos = (self.agent_start_pos)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        # neutral_reward = -0.01
        neutral_reward = 0
        agent_pos = self.agent_pos
        cur_object = self.grid.get(agent_pos[0], agent_pos[1])
        if cur_object:
            if cur_object.type == "ball":
                if cur_object.color == self.bad:
                    return -1
                else:
                    return neutral_reward
            elif cur_object.type == "goal":
                return (1 - 0.9 * (self.step_count / self.max_steps))

        return neutral_reward

class LivingRoomEnv5x5(LivingRoomEnv):
    def __init__(self, scene=0):
        super().__init__(scene=scene, size=5)


class LivingRoomEnv6x6(LivingRoomEnv):
    def __init__(self, scene=0, bad='purple'):
        super().__init__(scene=scene, size=6, bad=bad)


class LivingRoomEnv8x8(LivingRoomEnv):
    def __init__(self, scene=0, bad='purple'):
        super().__init__(scene=scene, size=8,  bad=bad)

class LivingRoomEnv9x9(LivingRoomEnv):
    def __init__(self, scene=4, bad='purple'):
        super().__init__(scene=scene, size=9,  bad=bad)

class LivingRoomEnv16x16(LivingRoomEnv):
    def __init__(self, scene=0, bad='purple'):
        super().__init__(scene=scene, size=16, bad=bad)


register(id='MiniGrid-LivingRoomEnv5x5-v0', entry_point='src.gym_minigrid.envs:LivingRoomEnv5x5')
register(id='MiniGrid-LivingRoomEnv6x6-v0', entry_point='src.gym_minigrid.envs:LivingRoomEnv6x6')
register(id='MiniGrid-LivingRoomEnv8x8-v0', entry_point='src.gym_minigrid.envs:LivingRoomEnv8x8')
register(id='MiniGrid-LivingRoomEnv9x9-v0', entry_point='src.gym_minigrid.envs:LivingRoomEnv9x9')
register(id='MiniGrid-LivingRoomEnv16x16-v0', entry_point='src.gym_minigrid.envs:LivingRoomEnv16x16')
