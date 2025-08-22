import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

register(
    id='Juggler-v0',
    entry_point='Juggler_gym:Juggler',
)


class Juggler(gym.Env):
    DT = 0.01 # time between frames
    GRAVITY = -500 # downwards acceleration (should be negative)
    #BALL_SPEED = 3 # TODO remove this, should be as quick as the hand?
    SHOULDER_RANGE = np.radians(5) # maximal degrees # TODO not needed currently because fixed
    SHOULDER_L_POS = np.array([150, 250])
    SHOULDER_R_POS = np.array([250, 250])
    UPPERARM_LENGTH = 90
    LOWERARM_LENGTH = 30
    CATCH_TOLERANCE = 10
    BALL_RADIUS = 10 # for collision detection

    # height of highest point of the parabola that the ball should pass through
    APEX_HEIGHT = lambda height: 100 * np.exp(height)
    APEX_TOLERANCE = 0.5 # radius of area aronud apex where throw should pass through


    def __init__(self, pattern, verbose=True, render_mode="rgb_array"):
        """
        Note that pattern defines how the environment behaves,
        e.g. it is crucial for the reward function (for checking if the right ball is caught),
        therefore it's shared between environment and agent.
        """
        # set pattern / task
        self.pattern = pattern
        self.period = len(self.pattern)
        self.n_balls = sum(self.pattern) // self.period # TODO error if not divisible

        self.state_dim = 6 + 4 * self.n_balls # (hold + elbow angle + elbow velocity) * 2 sides + (2 coordinates + 2 velocities) * number_of_balls
        self.action_dim = 4
        
        self.beats = 0
        self.catches = 0
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1, -1, 0, 0, -10, -10] + self.n_balls * [0, 0, -100, -100]),
                                            high=np.array([1, 1, 2*np.pi, 2*np.pi, 10, 10] + self.n_balls * [400, 600, 100, 100]),
                                            shape=(self.state_dim,), dtype=np.float32)

        self.verbose = verbose
        self.render_mode = render_mode
        self.screen = pygame.display.set_mode((400, 600))
        self.clock = pygame.time.Clock()
        self.isopen = True


    def reset(self, *, seed=None, options=None):
        """
        Resets the state and returns observations.
        """
        super().reset(seed=seed)
        
        # body state (4 degrees of freedom)
        self.hold_l = True
        self.hold_r = True
        self.elbow_l = np.pi
        self.elbow_r = np.pi / 4 # TODO initially shifted by 2*pi / len(pattern)
        self.d_elbow_l = 0
        self.d_elbow_r = 0
        # TODO fix shoulders? (make them static and not controllable)
        self.shoulder_l = np.radians(10)
        self.shoulder_r = np.radians(-10)

        # compute joint positions
        self._update_joints()

        # ball state
        self.time = 0 # absolute time
        self.beat = 0 # throw counter # TODO account for 0 and 2, which are not classical throws
        self.throws = 0 # throw counter
        self.catches = 0 # catch counter
        self.balls = []
        self._init_balls() # TODO can we change pattern of juggler after __init__?

        if self.render_mode == "rgb_array":
            self.render()
        return self._get_observations(), {"catches": self.catches, "beats": self.beats}


    def step(self, action):
        """
        Apply continuous forces to control body state.
        Continuous controls are accelerations, i.e. they smoothly change the state.
        TODO scaled to [-1, 1], such that the controller can be agnostic about their range.
        Returns observations including body and ball positions TODO as well as reward.
        """
        # update discrete body state
        self.hold_l, self.hold_r = action[:2] > 0 # TODO make it sigmoid
        cont_controls = action[2:]

        # increment beat if elbow crosses pi (only after first throw)
        # if self.throws > 0 and self.elbow_l < np.pi and self.elbow_l + self.d_elbow_l * Juggler.DT > np.pi:
        #     self.beat += 1
        #     print("beat", self.beat)
        # if self.throws > 0 and self.elbow_r < np.pi and self.elbow_r + self.d_elbow_r * Juggler.DT > np.pi:
        #     self.beat += 1
        #     print("beat", self.beat)

        # update continuous body state (elbow angle / angular velocity)
        self.d_elbow_l += cont_controls[0] * Juggler.DT
        self.d_elbow_r += cont_controls[1] * Juggler.DT
        self.elbow_l   += self.d_elbow_l   * Juggler.DT
        self.elbow_r   += self.d_elbow_r   * Juggler.DT
        self.elbow_l %= 2*np.pi
        self.elbow_r %= 2*np.pi

        self._update_joints()

        # simulate balls
        self._simulate_balls()
        terminate = self._check_drop() or self._check_collision()
        self.time += Juggler.DT

        # get_reward
        reward = self._get_reward()
        obs = self._get_observations()
        info = {"catches": self.catches, "throws": self.throws, "beats": self.beats, "time": self.time}
        return obs, reward, terminate, False, info


    def close(self):
        if self.render_mode == "rgb_array":
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


    def _get_observations(self):
        """
        The observations are body states and ball states.
        """
        obs = [self.hold_l, self.hold_r, self.elbow_l, self.elbow_r, self.d_elbow_l, self.d_elbow_r]
        for ball in self.balls:
            obs += [ball["pos"][0], ball["pos"][1], ball["vel"][0], ball["vel"][1]]
        return obs


    def _get_reward(self):
        # reward for turning in the right direction
        turning_reward = ((self.d_elbow_l > 0) + (self.d_elbow_r > 0)) * 0.01 # TODO
        # reward for flight
        dist = 0
        for ball in self.balls:
            if ball["dwell"] == -1: # if ball flying
                height = ball["height"]
                target = ball["target"]
                target_hand_pos = np.array([self.hand_l_pos, self.hand_r_pos][target])
                target_elbow_pos = np.array([self.elbow_l_pos, self.elbow_r_pos][target])
                apex = [1 if height % 2 == 1 else target_elbow_pos[0], Juggler.APEX_HEIGHT(height)]
                before = ball["pos"][0] < apex[0] if target == 1 else ball["pos"][0] > apex[0]
                if before: # if ball "before" apex: dist = distance(ball, apex) + distance(apex, elbow)
                    d = np.linalg.norm(ball["pos"] - apex) + np.linalg.norm(apex - target_elbow_pos)
                else: # if ball "behind" apex: dist = distance(ball, hand)
                    d = np.linalg.norm(ball["pos"] - target_hand_pos)
                dist += d
        fly_reward = 1/dist if dist != 0 else 0
        return turning_reward + fly_reward


    def _update_joints(self):
        # TODO fix shoulders?
        #self.shoulder_l, self.shoulder_r = np.radians(10), np.radians(-10)
        self.elbow_l_pos = -np.array([np.sin(self.shoulder_l), np.cos(self.shoulder_l)]) * Juggler.UPPERARM_LENGTH + Juggler.SHOULDER_L_POS
        self.elbow_r_pos = -np.array([np.sin(self.shoulder_r), np.cos(self.shoulder_r)]) * Juggler.UPPERARM_LENGTH + Juggler.SHOULDER_R_POS

        # update positions of elbow and hands
        # NOTE left arm (seen from front) turns counter-clockwise, right arm turns clockwise, zero degrees are pointing to the inside
        self.hand_l_pos = np.array([ np.cos(self.elbow_l), np.sin(self.elbow_l)]) * Juggler.LOWERARM_LENGTH + self.elbow_l_pos
        self.hand_r_pos = np.array([-np.cos(self.elbow_r), np.sin(self.elbow_r)]) * Juggler.LOWERARM_LENGTH + self.elbow_r_pos

        # compute hand tangential velocity (which balls take over when thrown)
        self.hand_l_vel = np.array([-np.sin(self.elbow_l), np.cos(self.elbow_l)]) * Juggler.LOWERARM_LENGTH * self.d_elbow_l
        self.hand_r_vel = np.array([ np.sin(self.elbow_r), np.cos(self.elbow_r)]) * Juggler.LOWERARM_LENGTH * self.d_elbow_r


    def _init_balls(self):
        """
        Init balls.
        All balls are placed in the two hands according to the pattern.
        Zero-throws do not require a ball.
        """
        n_placed = 0
        for beat, height in enumerate(self.pattern):
            if height == 0 or n_placed == self.n_balls: # only n_balls and no zero-throws
                continue
            # if non-zero throw, place in alternating hands
            origin = beat % 2
            ball = {
                "id": n_placed,
                "pos": [self.hand_l_pos, self.hand_r_pos][origin],
                "vel": [0, 0],
                "beat": beat, # in how many beats the ball will be thrown
                "dwell":  origin, # in which hand the ball dwells, -1 if in air
                "origin": origin, # hand in which the ball starts
                "height": height,
                "target": origin if height % 2 == 0 else 1-origin,
                "color": [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][n_placed % 5]
            }
            n_placed += 1
            self.balls += [ball]


    def _throw_ball(self, ball):
        """
        Release ball from hand.
        Every throw increases beat.
        """
        ball["height"] = self.pattern[self.beat % self.period]
        ball["beat"] += ball["height"]
        self.beat += 1

        if ball["height"] == 0: # TODO where to handle zeros, they don't belong to a ball but beat has to be counted
            return # zeros are not thrown
        
        self.throws += 1 # only for non-zero throws

        ball["origin"] = ball["dwell"]
        ball["target"] = ball["origin"] if ball["height"] % 2 == 0 else 1-ball["origin"]
        ball["dwell"] = -1 # in air
        if self.verbose:
            print("throw ball", ball["id"], "at height", ball["height"], "from hand", ball["origin"])
            print(ball)


    def _fly_ball(self, ball):
        # Euler integration for position and velocity
        ball["vel"] += np.array([0, Juggler.GRAVITY]) * Juggler.DT
        ball["pos"] += ball["vel"] * Juggler.DT


    def _try_catch_ball(self, ball):
        """
        Catch ball if
        1. target hand is in catching state (hold == True)
        2. ball is within CATCH_TOLERANCE distance of target hand
        3. number of beats that have passed equals ball height # TODO this is not the case for very long or very short dwell times
        """
        target = ball["target"]
        hold = [self.hold_l, self.hold_r][target]
        if not hold:
            return
        target_pos = [self.hand_l_pos, self.hand_r_pos][target]
        dist = np.linalg.norm(ball["pos"] - target_pos)
        if dist > Juggler.CATCH_TOLERANCE:
            return
        if self.beat >= ball["beat"] - 1: # ball should be caught 1 beat before next throw
            ball["dwell"] = target
            self.catches += 1
            if self.verbose:
                print("catch ball", ball["id"], "in hand", ball["target"])


    def _simulate_balls(self):
        """
        Simulate the ball movement.
        In dwell time, ball just remains in hand.
        If hand opened, ball takes over tangential velocity and flies in a parabola.
        If ball in reach for catching (and hand in catching state), ball snaps to hand.
        """
        for ball in self.balls:
            # keeping the ball in hand
            if self.hold_l and ball["dwell"] == 0:
                ball["pos"] = self.hand_l_pos
                ball["vel"] = self.hand_l_vel
            elif self.hold_r and ball["dwell"] == 1:
                ball["pos"] = self.hand_r_pos
                ball["vel"] = self.hand_r_vel

            # throwing
            # theoretically allows multiplex but TODO in notation
            if ball["dwell"] >= 0 and ball["beat"] == self.beat:
                active_hand = self.beat % 2
                if (active_hand == 0 and not self.hold_l) \
                or (active_hand == 1 and not self.hold_r):
                    self._throw_ball(ball)
            
            # if ball not in hand, let it fly and try to catch
            if ball["dwell"] < 0:
                self._fly_ball(ball)
                self._try_catch_ball(ball)


    def _check_drop(self):
        for ball in self.balls:
            if ball["pos"][1] < 0:
                if self.verbose:
                    print("drop", ball["id"])
                return True
        return False


    def _check_collision(self):
        """
        If balls collide in midair, episode should end.
        """
        for i in range(self.n_balls):
            for j in range(i+1, self.n_balls):
                if self.balls[i]["dwell"] == -1 and self.balls[j]["dwell"] == -1: # both balls flying
                    if np.linalg.norm(self.balls[i]["pos"] - self.balls[j]["pos"]) < Juggler.BALL_RADIUS:
                        if self.verbose:
                            print("collision", self.balls[i]["id"], self.balls[j]["id"])
                        return True
        return False

    
    def render(self, filename=None, show=False):
        if self.render_mode is None:
            return

        WIDTH, HEIGHT = 400, 600

        def to_pygame_coords(pos):
            # pos: (x, y) as numpy array or tuple
            x, y = pos
            return int(x), int(HEIGHT - y)

        self.surf = pygame.Surface((WIDTH, HEIGHT))
        self.surf.fill((255, 255, 255)) # white background

        # draw the body
        body_points = [
            to_pygame_coords((175, 125)),
            to_pygame_coords((225, 125)),
            to_pygame_coords(Juggler.SHOULDER_R_POS),
            to_pygame_coords(Juggler.SHOULDER_L_POS)
        ]
        pygame.draw.polygon(self.surf, color=(215, 215, 255), points=body_points)
        pygame.draw.circle(self.surf, color=(215, 215, 255), center=to_pygame_coords((200, 300)), radius=30)

        # draw the upper arms
        pygame.draw.line(self.surf, color=(0, 0, 0), start_pos=to_pygame_coords(Juggler.SHOULDER_L_POS), end_pos=to_pygame_coords(self.elbow_l_pos), width=2)
        pygame.draw.line(self.surf, color=(0, 0, 0), start_pos=to_pygame_coords(Juggler.SHOULDER_R_POS), end_pos=to_pygame_coords(self.elbow_r_pos), width=2)

        # draw the lower arms
        pygame.draw.line(self.surf, color=(0, 0, 0), start_pos=to_pygame_coords(self.elbow_l_pos), end_pos=to_pygame_coords(self.hand_l_pos), width=2)
        pygame.draw.line(self.surf, color=(0, 0, 0), start_pos=to_pygame_coords(self.elbow_r_pos), end_pos=to_pygame_coords(self.hand_r_pos), width=2)

        # draw the balls
        for ball in self.balls:
            pygame.draw.circle(self.surf, color=ball["color"], center=to_pygame_coords(ball["pos"]), radius=Juggler.BALL_RADIUS)

        # draw the apex
        for ball in self.balls:
            if ball["dwell"] == -1: # if flying
                height = ball["height"]
                target = ball["target"]
                target_elbow_pos = np.array([self.elbow_l_pos, self.elbow_r_pos][target])
                midpoint = np.mean(self.SHOULDER_L_POS[0], self.SHOULDER_R_POS[0])
                apex = [midpoint if height % 2 == 1 else target_elbow_pos[0], Juggler.APEX_HEIGHT(height)]
                pygame.draw.circle(self.surf, color=ball["color"], center=to_pygame_coords(apex), radius=5)

        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(20)  # Limit to 30 FPS



class OptimalAgent():
    """
    The OptimalAgent knows for each desired throw/height, 
    at which angle and with which speed to throw.
    """
    AVG_SPEED = 3 # average speed when not throwing
    FACTOR = 3
    # angle at which to throw: odd throws at negative angles (below horizontal), even at positive (above horizontal)
    height2theta = np.array([
        0,
        2-1/3,
        0+1/10,
        2-1/10,
        0+1/6,
        2-1/6,
    ]) * np.pi
    # speed with which to throw
    height2omega = np.array([
        AVG_SPEED,
        10,
        AVG_SPEED, # for passive 2
        15,
        15,
        25,
    ])
    #THROW_TOLERANCE = 0.2 # radians

    def __init__(self, pattern):
        self.pattern = pattern
        self.DT = 0.05 # TODO can be quicker than Juggler.DT
        self.time = 0
        self.beat = 0 # TODO


    def act(self, observations, info):
        # print(observations)
        elbow_l, elbow_r, d_elbow_l, d_elbow_r = observations[2:6]
        #print(d_elbow_l)

        # current beat
        self.beat = info["beats"]
        throw_hand = self.beat % 2

        # current throw
        height = self.pattern[self.beat % len(self.pattern)]
        theta = OptimalAgent.height2theta[height]
        omega = OptimalAgent.height2omega[height]

        # adapt speed
        if OptimalAgent._radial_diff(elbow_l, theta) < np.pi: # accelerate in the hemi-circle before the throw
            dd_elbow_l = self.FACTOR * (omega - d_elbow_l)
        else: # decelerate in hemi-circle after the throw
            dd_elbow_l = self.FACTOR * (self.AVG_SPEED - d_elbow_l)
        if elbow_r < theta: # accelerate
            dd_elbow_r = self.FACTOR * (omega - d_elbow_r)
        else: # decelerate in hemi-circle after the throw
            dd_elbow_r = self.FACTOR * (self.AVG_SPEED - d_elbow_r)

        # constant velocity after initial acceleration
        #dd_elbow_l = 20 - d_elbow_l
        #dd_elbow_r = 20 - d_elbow_r

        # open hands shortly before theta and close shortly after
        dist_l = OptimalAgent._radial_dist(theta, elbow_l)
        hold_l = dist_l > (d_elbow_l + dd_elbow_l * self.DT) * self.DT
        dist_r = OptimalAgent._radial_dist(theta, elbow_r)
        hold_r = dist_r > (d_elbow_r + dd_elbow_r * self.DT) * self.DT

        return np.array([hold_l, hold_r, dd_elbow_l, dd_elbow_r])


    @staticmethod
    def _radial_diff(a, b):
        """
        The difference from a to b in anti-clockwise direction.
        where a, b are angles in [0, 2pi]
        """
        if b >= a:
            return b - a
        else:
            return 2*np.pi - a + b


    @staticmethod
    def _radial_dist(a, b):
        """
        The distance between angles a and b on the circle.
        """
        dist = np.abs(a-b)
        if dist > np.pi:
            dist = 2*np.pi - dist
        return dist


if __name__ == "__main__":
    PATTERN = [3,0,0] # [4,4,4,4]
    N_STEP = 1000

    env = Juggler(PATTERN)
    agent = OptimalAgent(PATTERN)

    terminate = False
    rewards = []
    step = 0
    obs, info = env.reset()
    while not terminate and step < N_STEP:
        ctrl = agent.act(obs, info)
        obs, reward, terminate, _, info = env.step(ctrl)
        rewards += [reward]
        #print(reward)
        env.render()
        step += 1
