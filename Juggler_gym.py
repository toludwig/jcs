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
    GRAVITY = -9.81 # downwards acceleration (should be negative)
    BALL_SPEED = 1 # TODO remove this, should be as quick as the hand?
    SHOULDER_RANGE = np.radians(5) # maximal degrees # TODO not needed currently because fixed
    SHOULDER_L_POS = np.array([150, 250])
    SHOULDER_R_POS = np.array([250, 250])
    UPPERARM_LENGTH = 90
    LOWERARM_LENGTH = 30
    CATCH_TOLERANCE = 10
    BALL_RADIUS = 10 # for collision detection

    # height of highest point of the parabola that the ball should pass through
    APEX_HEIGHT = lambda height: 0.5 + height / 3
    APEX_TOLERANCE = 0.5 # radius of area aronud apex where throw should pass through


    def __init__(self, pattern, verbose=True, render_mode=None):
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
        self.observation_space = spaces.Box(low=np.array([-1, -1, 0, 0,-10, -10] + self.n_balls * [-2, -2, -10, -10]),
                                            high=np.array([1, 1, 2*np.pi, 2*np.pi, 10, 10] + self.n_balls * [4, 10, 10, 10], dtype=np.float32),
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
        self.elbow_l = 0
        self.elbow_r = 2 * np.pi / 3 # TODO initially shifted by 2*pi / len(pattern)
        self.d_elbow_l = 30
        self.d_elbow_r = 30
        # TODO fix shoulders? (make them static and not controllable)
        self.shoulder_l = np.radians(10)
        self.shoulder_r = np.radians(-10)

        # compute joint positions
        self._update_joints()

        # ball state
        self.time = 0 # absolute time
        self.beat = 0 # throw counter # TODO account for 0 and 2, which are not classical throws
        self.catches = 0 # catch counter
        self.balls = []
        self._init_balls() # TODO can we change pattern of juggler after __init__?

        if self.render_mode == "human":
            self.render()
        return self._get_observations(), {"catches": self.catches, "beats": self.beats}


    def sample_action(self):
        """
        Samples a random action where the first two are discrete (boolean)
        and the second two are continuous (in range [-1,1]).
        """
        #disc = np.random.choice([0,1], size=2)
        #cont = np.random.rand(2) * 2 - 1
        #return np.hstack([disc, cont])
        return np.random.rand(4) * 2 - 1


    def sample_state(self):
        """
        Sample a random state. See _get_observation.
        """
        hold = np.random.choice([0,1], size=2)        
        elbow = np.random.rand(4) * 2*np.pi # 4 dims for 2 pos and 2 vel
        balls = []
        for _ in range(self.n_balls):
            balls += [np.random.rand(4) * 2*np.pi] # 4 dims for 2 pos and 2 vel
        counters = np.random.randint(0, 10, size=2) # 2 more dims for throw and catch counter
        return np.hstack([hold, elbow] + balls + [counters]) 


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

        # update continuous body state (elbow angle / angular velocity)
        self.d_elbow_l += cont_controls[0] * Juggler.DT
        self.d_elbow_r += cont_controls[1] * Juggler.DT
        self.elbow_l   += self.d_elbow_l   * Juggler.DT
        self.elbow_r   += self.d_elbow_r   * Juggler.DT

        # increase beat count with every zero crossing of either elbow
        if self.elbow_l > 2*np.pi:
            self.beat += 1
            if self.verbose:
                print("beat", self.beat)
            self.elbow_l %= 2*np.pi
        if self.elbow_r > 2*np.pi:
            self.beat += 1
            if self.verbose:
                print("beat", self.beat)
            self.elbow_r %= 2*np.pi

        self._update_joints()

        # simulate balls
        self.time += Juggler.DT
        self._simulate_balls()
        terminate = self._check_drop() or self._check_collision()

        # get_reward
        reward = self._get_reward()

        # observations
        obs = self._get_observations()
        return obs, reward, terminate, False, {"catches": self.catches, "beats": self.beats}


    def close(self):
        if self.render_mode == "human":
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


    def _get_observations(self):
        """
        The observations are body states and ball states
        plus the number of throws (beat) and catches.
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
        # NOTE left arm (seen from front) turns counter-clockwise, right arm turns clockwise, zero degrees are at 6 o'clock
        self.hand_l_pos = np.array([ np.sin(self.elbow_l), -np.cos(self.elbow_l)]) * Juggler.LOWERARM_LENGTH + self.elbow_l_pos
        self.hand_r_pos = np.array([-np.sin(self.elbow_r), -np.cos(self.elbow_r)]) * Juggler.LOWERARM_LENGTH + self.elbow_r_pos

        # compute hand tangential velocity (which balls take over when thrown)
        self.hand_l_vel = np.array([ np.cos(self.elbow_l), np.sin(self.elbow_l)]) * Juggler.LOWERARM_LENGTH
        self.hand_r_vel = np.array([-np.cos(self.elbow_r), np.sin(self.elbow_r)]) * Juggler.LOWERARM_LENGTH


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
                "vel": [self.hand_l_vel, self.hand_r_vel][origin],
                "beat": beat,
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
        ball["origin"] = ball["dwell"]
        ball["target"] = ball["origin"] if ball["height"] % 2 == 0 else 1-ball["origin"]
        ball["dwell"] = -1 # in air
        #ball["beat"] = self.beat
        print("ball_vel", ball["vel"])
        print("hand_l_vel", self.hand_l_vel)
        print("hand_r_vel", self.hand_r_vel)
        if self.verbose:
            print("throw ball", ball["id"], "at height", ball["height"], "from hand", ball["origin"])


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
        beats = self.beat - ball["beat"] # beats while in air
        if beats == ball["height"] - 1:
            ball["beat"] += beats + 1
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
                ball["vel"] = self.hand_l_vel * self.BALL_SPEED
            elif self.hold_r and ball["dwell"] == 1:
                ball["pos"] = self.hand_r_pos
                ball["vel"] = self.hand_r_vel * self.BALL_SPEED

            # when hand with ball opens, throw top ball of that hand
            if (not self.hold_l and ball["dwell"] == 0) \
            or (not self.hold_r and ball["dwell"] == 1):
                if ball["beat"] == self.beat: # if multiple balls in hand (at start), throw the right one
                    self._throw_ball(ball)
            
            # if ball not in hand, let it fly
            if ball["dwell"] < 0:
                self._fly_ball(ball)

                # try and catch ball
                self._try_catch_ball(ball)


    def _check_drop(self):
        for ball in self.balls:
            if ball["pos"][1] < -2:
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
                apex = [1 if height % 2 == 1 else target_elbow_pos[0], Juggler.APEX_HEIGHT(height)]
                pygame.draw.circle(self.surf, color=ball["color"], center=to_pygame_coords(apex), radius=5)

        self.screen.blit(self.surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS



class OptimalAgent():
    """
    The OptimalAgent knows for each desired throw/height, 
    at which angle and with which speed to throw.
    """
    # TODO angle at which to throw
    height2theta = {
        0: 0,
        1: np.pi/6,
        2: np.nan,
        3: np.pi/3,
        4: 11*np.pi/20,
        5: 7*np.pi/16,
        6: 16*np.pi/30
    }
    # TODO speed with which to throw
    height2omega = {
        0: 0,
        1: 1,
        2: 0,
        3: 3,
        4: 4,
        5: 5,
        6: 6
    }

    #THROW_TOLERANCE = 0.2 # radians

    def __init__(self, pattern):
        self.pattern = pattern
        self.DT = 0.05 # TODO can be quicker than Juggler.DT
        self.time = 0
        self.beat = 0 # TODO


    def act(self, observations, info):
        # print(observations)
        elbow_l, elbow_r, d_elbow_l, d_elbow_r = observations[2:6]

        # current beat
        self.beat = info["beats"]
        throw_hand = self.beat % 2

        # current throw
        height = self.pattern[self.beat % len(self.pattern)]
        theta = OptimalAgent.height2theta[height]
        omega = OptimalAgent.height2omega[height]

        # adapt speed
        # if elbow_l < theta: # accelerate
        #     dd_elbow_l = omega - d_elbow_l
        # elif elbow_l < 3/2*np.pi: # decelerate
        #     dd_elbow_l = 2 - d_elbow_l
        # else:
        #     dd_elbow_l = 0
        # if elbow_r < theta: # accelerate
        #     dd_elbow_r = omega - d_elbow_r
        # elif elbow_r < 3/2*np.pi: # decelerate
        #     dd_elbow_r = 2 - d_elbow_r
        # else:
        #     dd_elbow_r = 0

        # constant velocity
        dd_elbow_l = 30 - d_elbow_l
        dd_elbow_r = 30 - d_elbow_r

        # open hands for throw on the inside depending on throw angle
        # and close hand for catch on the outside (can catch with hand closed)
        hold_l = elbow_l < theta or np.pi*2/3 < elbow_l
        hold_r = elbow_r < theta or np.pi*2/3 < elbow_r

        # TODO keep track of time
        # self.time += self.DT

        return np.array([hold_l, hold_r, dd_elbow_l, dd_elbow_r])



if __name__ == "__main__":
    PATTERN = [5,3,1] # [4,4,4,4]
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
