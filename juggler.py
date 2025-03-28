import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
A simplified juggling simulation for vanilla site-swaps.
There are four degrees of freedom at the elbows and hands.
Two angular forces can "turn" the elbow,
and two discrete controls can open and close the hands.

Reward is calculated as the inverse distance of the thrown ball to the catching hand.

NOTE left and right are as seen from the front, not from the juggler's perspective
"""

class Juggler():
    DT = 0.01 # time between frames
    GRAVITY = -2
    BALL_SPEED = 5 # TODO remove this, should be as quick as the hand?
    ELBOW_SPEED = 3.8 # TODO rename?
    SHOULDER_RANGE = np.radians(5) # maximal degrees
    SHOULDER_L_POS = np.array([0, 2])
    SHOULDER_R_POS = np.array([2, 2])
    UPPERARM_LENGTH = 1.5
    LOWERARM_LENGTH = 0.5
    CATCH_TOLERANCE = 0.5

    def __init__(self, pattern, rendering=True, verbose=True):
        # body state (4 degrees of freedom)
        self.hold_l = True
        self.hold_r = True
        self.elbow_l = 0
        self.elbow_r = 2 * np.pi / 3 # TODO initially shifted by 2*pi / len(pattern)
        self.d_elbow_l = Juggler.ELBOW_SPEED
        self.d_elbow_r = Juggler.ELBOW_SPEED
        # TODO fix shoulders? (make them static and not controllable)
        self.shoulder_l = np.radians(10)
        self.shoulder_r = np.radians(-10)

        # compute joint positions
        self._update_joints()

        # ball state
        self.time = 0 # absolute time
        self.beat = 0 # beat is incremented with every catch and throw
        self.catches = 0 # catch counter
        self.balls = []
        self._init_pattern(pattern) # TODO can we change pattern of juggler after __init__?

        # drawing
        self.rendering = rendering
        self.frames = [] # for saving frames
        if rendering:
            self._draw_init()
        self.verbose = verbose


    def get_state(self):
        body = (self.hold_l, self.hold_r, self.elbow_l, self.elbow_r, self.d_elbow_l, self.d_elbow_r)
        return body, self.balls

    def get_reward(self):
        dist = 0
        for ball in self.balls:
            if ball["dwell"] == -1 and ball["vel"][1] < 0: # distance only when flying down
                dist += np.linalg.norm(ball["pos"] - [self.hand_l_pos, self.hand_r_pos][ball["target"]])
        return 1/dist if dist != 0 else 0
        #return 1 # time without drop

    def step(self, disc_controls, cont_controls):
        """
        Apply discrete inputs and continuous forces to control body state.
        Continuous controls are accelerations, i.e. they smoothly change the state.
        TODO scaled to [-1, 1], such that the controller can be agnostic about their range.
        Discrete controls are values to which the state is set immediately (not smoothly).
        Returns observations including body and ball positions TODO as well as reward.
        """
        # update discrete body state
        self.hold_l, self.hold_r = disc_controls

        # update continuous body state (elbow angle / angular velocity)
        self.d_elbow_l += cont_controls[0] * Juggler.DT
        self.d_elbow_r += cont_controls[1] * Juggler.DT
        self.elbow_l   += self.d_elbow_l   * Juggler.DT
        self.elbow_r   += self.d_elbow_r   * Juggler.DT
        self.elbow_l %= 2*np.pi
        self.elbow_r %= 2*np.pi
        self._update_joints()

        # simulate balls
        self.time += Juggler.DT
        self._simulate_balls()
        drop = self._check_drop()

        # get_reward
        reward = self.get_reward()

        # rendering
        if self.rendering:
            self._draw()

        # observations
        state = self.get_state()
        return state, drop, reward


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
        self.hand_l_vel = np.array([ np.cos(self.elbow_l), np.sin(self.elbow_l)]) * Juggler.LOWERARM_LENGTH * Juggler.BALL_SPEED
        self.hand_r_vel = np.array([-np.cos(self.elbow_r), np.sin(self.elbow_r)]) * Juggler.LOWERARM_LENGTH * Juggler.BALL_SPEED


    def _init_pattern(self, pattern):
        """
        Init pattern and balls.
        All balls are placed in the two hands but only the top ball is thrown.
        """
        # set pattern / task
        self.pattern = pattern
        self.period = len(self.pattern)
        self.n_balls = sum(self.pattern) // self.period # TODO error if not divisible

        # init all balls
        for bid in range(self.n_balls):
            origin = bid % 2
            # height = self.pattern[self.beat % self.period]
            # target = origin if height % 2 == 0 else 1-origin
            ball = {
                "id": bid,
                "pos": [self.hand_l_pos, self.hand_r_pos][origin],
                "vel": [self.hand_l_vel, self.hand_r_vel][origin],
                "dwell":  origin, # in which hand the ball dwells, -1 if in air
                "origin": origin, # hand in which the ball starts
            }
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
        ball["beat"] = self.beat # TODO needed?
        if self.verbose:
            print("id", ball["id"])
            print("beat", self.beat)
            print("origin", ball["origin"])

        self.beat += 1 # TODO after take_ball?


    def _fly_ball(self, ball):
        # Euler integration for position and velocity
        ball["vel"] += np.array([0, Juggler.GRAVITY]) * Juggler.DT
        ball["pos"] += ball["vel"] * Juggler.DT


    def _try_catch_ball(self, ball):
        """
        Catch ball if
        1. target hand is in catching state (hold == True)
        2. number of beats that have passed equals ball height
        3. ball is within CATCH_TOLERANCE distance of target hand
        """
        target = ball["target"]
        hold = [self.hold_l, self.hold_r][target]
        if not hold:
            return
        if self.verbose:
            print("beat", self.beat, "  ball beat", ball["beat"])
        beats = self.beat - ball["beat"] # beats between
        if beats != ball["height"]:
            return
        target_pos = [self.hand_l_pos, self.hand_r_pos][target]
        dist = np.linalg.norm(ball["pos"] - target_pos)
        if self.verbose:
            print(dist)
        if dist < Juggler.CATCH_TOLERANCE:
            ball["dwell"] = target
            self.beat += 1
            self.catches += 1
            if self.verbose:
                print("catch!")


    def _simulate_balls(self):
        """
        Simulate the ball movement.
        In dwell time, ball just remains in hand.
        If hand opened, ball takes over tangential velocity and flies in a parabola.
        If ball in reach for catching (and hand in catching state), ball snaps to hand.
        """
        for bid, ball in enumerate(self.balls):
            # keeping the ball in hand
            if self.hold_l and ball["dwell"] == 0:
                ball["pos"] = self.hand_l_pos
                ball["vel"] = self.hand_l_vel
            elif self.hold_r and ball["dwell"] == 1:
                ball["pos"] = self.hand_r_pos
                ball["vel"] = self.hand_r_vel

            # when hand with ball opens, throw top ball
            if (not self.hold_l and ball["dwell"] == 0) \
            or (not self.hold_r and ball["dwell"] == 1):
                if bid == self.beat % self.period: # if in order
                    if self.verbose:
                        print("throw ball", bid)
                        print("initial v:", ball["vel"])
                    self._throw_ball(ball)
            
            # if ball not in hand, let it fly
            if ball["dwell"] < 0:
                self._fly_ball(ball)

                if bid == 0:
                    if self.verbose:
                        # print(ball["vel"])
                        print(ball["pos"])

                # try and catch ball
                self._try_catch_ball(ball)


    def _check_drop(self):
        for ball in self.balls:
            if ball["pos"][1] < -2:
                return True
        return False


    def _draw(self):
        body = self._draw_body()
        arms = self._draw_arms()
        balls = self._draw_balls()
        self.frames += [body + arms + balls]

    def _draw_init(self):
        self.fig = plt.figure(figsize=(5,5))
        self.ax = plt.axes(xlim=[-2,4], ylim=[-2,4])
        self.ax.axis("off")

    def _draw_body(self):
        body, = self.ax.fill([Juggler.SHOULDER_L_POS[0], Juggler.SHOULDER_R_POS[0], 0.75*Juggler.SHOULDER_R_POS[0], 0.25*Juggler.SHOULDER_R_POS[0]],
                             [Juggler.SHOULDER_L_POS[1], Juggler.SHOULDER_R_POS[1], 0, 0], c="lightgrey", animated=True)
        head = self.ax.add_patch(plt.Circle((1, 2.6), 0.5, color="lightgrey", zorder=0))
        return [body, head]

    def _draw_arms(self):
        arm_l = np.vstack([Juggler.SHOULDER_L_POS, self.elbow_l_pos, self.hand_l_pos])
        arm_r = np.vstack([Juggler.SHOULDER_R_POS, self.elbow_r_pos, self.hand_r_pos])
        arm_l = self.ax.plot(arm_l[:,0], arm_l[:,1], c="b", animated=True)
        arm_r = self.ax.plot(arm_r[:,0], arm_r[:,1], c="b", animated=True)
        return arm_l + arm_r

    def _draw_balls(self):
        balls = []
        for i, ball in enumerate(self.balls):
            pos = ball["pos"]
            balls += self.ax.plot(pos[0], pos[1], "ro", markersize=20, animated=True)
        return balls
    
    def render(self, filename=None):
        self.ani = animation.ArtistAnimation(self.fig, self.frames, interval=Juggler.DT*1000, blit=True)
        writergif = animation.PillowWriter(fps=30) 
        if filename is not None:
            self.ani.save(f"{filename}.gif", writer=writergif)
        plt.show()


class OptimalAgent():
    """
    The OptimalAgent knows for each desired throw/height, 
    at which angle and with which speed to throw.
    """
    # TODO angle at which to throw
    height2theta = {1: np.pi/6,
                    2: np.nan,
                    3: np.pi/3, # TODO
                    4: 5*np.pi/8,
                    5: 3*np.pi/8,
                    6: 6*np.pi/10}
    # TODO speed with which to throw
    height2omega = {1: 1,
                    2: 0,
                    3: 3,
                    4: 4,
                    5: 5,
                    6: 6}

    #THROW_TOLERANCE = 0.2 # radians

    def __init__(self, pattern):
        self.pattern = pattern
        self.DT = 0.05 # TODO can be quicker than Juggler.DT
        self.time = 0
        self.beat = 0 # TODO


    def control(self, state):
        (body, balls) = state # TODO use balls
        _, _, elbow_l, elbow_r, d_elbow_l, d_elbow_r = body

        # current beat
        if elbow_l == 0 or elbow_r == 0: # when arm crosses 0, start new beat
            self.beat += 1
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

        dd_elbow_l = 0
        dd_elbow_r = 0

        # TODO adapt speed of catching hand
        if throw_hand == 0: # catch_hand == 1
            dd_elbow_r = 0
        else:
            dd_elbow_l = 0

        # TODO fix shoulders?
        # shoulder_l = np.sin(time) * Juggler.SHOULDER_RANGE + np.radians(10)
        # shoulder_r = np.cos(time) * Juggler.SHOULDER_RANGE + np.radians(-10)

        # open hands for throw on the inside depending on throw angle
        # and close hand for catch on the outside (can catch with hand closed)
        hold_l = elbow_l < theta or np.pi/2 < elbow_l
        hold_r = elbow_r < theta or np.pi/2 < elbow_r

        # TODO keep track of time
        # self.time += self.DT

        disc_controls = (hold_l, hold_r)
        cont_controls = (dd_elbow_l, dd_elbow_r)
        return disc_controls, cont_controls



if __name__ == "__main__":
    PATTERN = [3,3,3]
    N_STEP = 350

    env = Juggler(PATTERN, rendering=True)
    agent = OptimalAgent(PATTERN)

    ctrl = agent.control(env.get_state())
    drop = False
    rewards = []
    step = 0
    while not drop and step < N_STEP:
        state, drop, reward = env.step(*ctrl)
        ctrl = agent.control(state)
        rewards += [reward]
        step += 1

    env.render() # str(PATTERN) + "uniform_speed.gif")
