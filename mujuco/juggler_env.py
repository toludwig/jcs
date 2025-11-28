"""
Gymnasium environment for the MuJoCo Juggler model.

This environment implements a 3D juggling task where a humanoid with two arms
must catch and throw three balls using weld constraints for catching.

Action Space: 10-dimensional continuous
    - Actions 0-7: Motor torques for 8 arm joints (shoulder1, shoulder2, elbow, wrist for each arm)
    - Actions 8-9: Grip controls for right and left hands ([-1, 1] mapped to [0, 1] for grip strength)

Observation Space: Variable size
    - Joint positions (qpos) and velocities (qvel)
    - Palm positions and velocities (both hands)
    - Ball positions and velocities (3 balls)
    - Touch sensor readings (6 sensors: 3 balls × 2 hands)
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


def quat_conj(q):
    """Conjugate of quaternion q = (w, x, y, z)"""
    qc = q.copy()
    qc[1:] *= -1
    return qc


def quat_mul(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_rotate(q, v):
    """Rotate vector v by unit quaternion q"""
    qv = np.array([0.0, *v], dtype=np.float64)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def world_to_local(p_world, frame_pos, frame_quat):
    """Convert world position to local frame coordinates"""
    return quat_rotate(quat_conj(frame_quat), (p_world - frame_pos))


def rel_pose(body1_pos, body1_quat, body2_pos, body2_quat):
    """Compute relative pose of body2 in body1's frame"""
    relpos = world_to_local(body2_pos, body1_pos, body1_quat)
    relquat = quat_mul(quat_conj(body1_quat), body2_quat)
    return relpos, relquat / np.linalg.norm(relquat)


class JugglerEnv(gym.Env):
    """
    MuJoCo Juggler Environment
    
    A humanoid with two arms must learn to juggle three balls by:
    - Controlling 8 arm joints (shoulders, elbows, wrists)
    - Opening/closing hands to catch and release balls
    - Using weld constraints to "grip" balls when caught
    
    Attributes:
        xml_path (str): Path to the MuJoCo XML model file
        frame_skip (int): Number of simulation steps per environment step
        render_mode (str): Rendering mode ('rgb_array', 'human', or None)
    """
    
    metadata = {
        "render_modes": ["rgb_array", "human"],
        "render_fps": 60,
    }
    
    def __init__(
        self,
        xml_path=None,
        render_mode=None,
        frame_skip=10,
        img_size=(640, 480),
        reward_weights=None,
    ):
        """
        Initialize the Juggler environment.
        
        Args:
            xml_path: Path to juggler.xml file. If None, uses default path.
            render_mode: Rendering mode ('rgb_array', 'human', or None)
            frame_skip: Number of simulation steps per environment step
            img_size: Image size for rendering (width, height)
            reward_weights: Dictionary with reward component weights
        """
        super().__init__()
        
        # Find XML file path
        if xml_path is None:
            # Try to find it relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(current_dir, "assests", "juggler.xml")
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found: {xml_path}")
        
        self.xml_path = xml_path
        self.frame_skip = int(frame_skip)
        self.render_mode = render_mode
        self.img_size = img_size
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize renderer if needed
        self.renderer = None
        if render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, width=img_size[0], height=img_size[1])
        elif render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer_sync = True
        
        # Get body IDs
        self.hand_r_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_right")
        self.hand_l_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand_left")
        self.ball_bids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball1")
        ]
        
        # Get equality constraint IDs for catching
        self.eq_ids = {}
        for hand in ["right", "left"]:
            eq_name = f"catch_{hand}_ball1"
            eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
            self.eq_ids[(hand, "ball1")] = eq_id
        
        # Get sensor IDs (touch sensors)
        self.touch_sids = []
        for hand in ["right", "left"]:
            sensor_name = f"touch_{hand}_ball1"
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            self.touch_sids.append(sid)
        
        # Get actuator IDs
        self.actuator_names = [
            "shoulder1_right", "shoulder2_right", "elbow_right", "wrist_right",
            "shoulder1_left", "shoulder2_left", "elbow_left", "wrist_left",
        ]
        self.act_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.actuator_names
        ]
        
        # Action space: 8 motors + 2 grips
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )
        
        # Observation space: qpos + qvel + palms + balls + touch sensors
        obs_dim = self.model.nq + self.model.nv  # base state
        obs_dim += (3 + 3) * 2  # 2 palms (pos + vel)
        obs_dim += (3 + 3) * 1  # 1 ball (pos + vel)
        obs_dim += 2  # touch sensors (2 sensors for 1 ball)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # State tracking
        self.grip_right = 0.0
        self.grip_left = 0.0
        self.attached = {"right": None, "left": None}
        
        # Reward weights
        self.reward_weights = reward_weights or {
            "height": 0.5,
            "distance": -0.1,
            "touch": 0.2,
            "catch": 1.0,
            "survive": 0.01,
        }
        
        self.control_timestep = self.model.opt.timestep * self.frame_skip
        
    def _body_pose(self, body_id):
        """Get body position and quaternion"""
        pos = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()
        return pos, quat
    
    def _body_vel(self, body_id):
        """Get body linear velocity in world frame"""
        v_local = self.data.cvel[body_id, :3]  # linear vel in body frame
        q = self.data.xquat[body_id]
        v_world = quat_rotate(q, v_local)
        return v_world
    
    def _site_worldpos(self, site_name):
        """Get site position in world frame"""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[site_id].copy()
    
    def _set_weld_to_current_pose(self, eq_id, parent_bid, child_bid):
        """Configure weld constraint to current relative pose"""
        parent_pos, parent_quat = self._body_pose(parent_bid)
        child_pos, child_quat = self._body_pose(child_bid)
        relpos, relquat = rel_pose(parent_pos, parent_quat, child_pos, child_quat)
        
        # In MuJoCo, eq_data is a 2D array: (neq, nv) where nv is the number of values per constraint
        # For weld constraints, first 3 values are position, next 4 are quaternion
        # eq_data[eq_id] gives us the row for this constraint
        try:
            # Set position (first 3 values) and quaternion (next 4 values)
            self.model.eq_data[eq_id, 0:3] = relpos
            self.model.eq_data[eq_id, 3:7] = relquat
        except (AttributeError, ValueError, TypeError) as e:
            # If model.eq_data is read-only or shape is different, try alternative approach
            # Copy the data, modify it, and reassign
            print(f"Error setting weld data: {e}")
            eq_data_copy = self.model.eq_data.copy()
            eq_data_copy[eq_id, 0:3] = relpos
            eq_data_copy[eq_id, 3:7] = relquat
            self.model.eq_data[:] = eq_data_copy
    
    def _activate_weld(self, eq_id, activate=True):
        """Activate or deactivate a weld constraint"""
        # In MuJoCo, equality constraints can be controlled through eq_active
        # Try data.eq_active first (runtime control in MuJoCo 2.3+)
        if hasattr(self.data, 'eq_active'):
            self.data.eq_active[eq_id] = 1 if activate else 0
        # Fallback: use model.eq_active0 (initial state array)
        elif hasattr(self.model, 'eq_active0'):
            self.model.eq_active0[eq_id] = 1 if activate else 0
        else:
            # If neither exists, try model.eq_active as fallback
            try:
                self.model.eq_active[eq_id] = 1 if activate else 0
            except AttributeError:
                raise AttributeError(
                    "Cannot find eq_active attribute in model or data. "
                    "Please check your MuJoCo version compatibility."
                )
        
        # Ensure forward dynamics is called to apply changes
        mujoco.mj_forward(self.model, self.data)
    
    def _try_catch(self, hand, ball_name):
        """Attempt to catch a ball with the specified hand"""
        if self.attached[hand] is not None:
            return False  # Already holding something
        
        eq_id = self.eq_ids[(hand, ball_name)]
        parent_bid = self.hand_r_bid if hand == "right" else self.hand_l_bid
        child_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ball_name)
        
        # Set weld to current relative transform and activate
        self._set_weld_to_current_pose(eq_id, parent_bid, child_bid)
        self._activate_weld(eq_id, True)
        self.attached[hand] = ball_name
        return True
    
    def _release(self, hand):
        """Release any ball held by the specified hand"""
        if self.attached[hand] is None:
            return
        
        ball = self.attached[hand]
        eq_id = self.eq_ids[(hand, ball)]
        self._activate_weld(eq_id, False)
        self.attached[hand] = None
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Torso is fixed (no joints), so no need to set qpos for it
        # Forward kinematics to get hand position
        mujoco.mj_forward(self.model, self.data)
        
        # Get hand position and palm site position
        hand_r_pos = self.data.xpos[self.hand_r_bid].copy()
        palm_r_pos = self._site_worldpos("palm_right")
        # Position ball at palm site (slightly forward from hand center for better visibility)
        ball1_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball1")
        ball1_qadr = self.model.jnt_qposadr[ball1_joint_id]
        # Set ball position (freejoint has 7 DOF: 3 translation + 4 quaternion)
        self.data.qpos[ball1_qadr:ball1_qadr+3] = palm_r_pos
        # Set ball orientation to identity quaternion (no rotation)
        self.data.qpos[ball1_qadr+3:ball1_qadr+7] = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Initialize ball with zero velocity (will be attached to hand)
        ball1_vadr = self.model.jnt_dofadr[ball1_joint_id]
        # Freejoint has 6 velocity DOF: 3 angular + 3 linear
        self.data.qvel[ball1_vadr:ball1_vadr+3] = [0.0, 0.0, 0.0]  # zero angular velocity
        self.data.qvel[ball1_vadr+3:ball1_vadr+6] = [0.0, 0.0, 0.0]  # zero linear velocity
        
        # Reset grip state - start with right hand gripping
        self.grip_right = 1.0
        self.grip_left = 0.0
        
        # Forward kinematics again with ball at correct position
        mujoco.mj_forward(self.model, self.data)
        
        # Deactivate weld first to reposition ball
        self._activate_weld(self.eq_ids[("right", "ball1")], False)
        self._activate_weld(self.eq_ids[("left", "ball1")], False)
        
        # Forward kinematics to settle
        mujoco.mj_forward(self.model, self.data)
        
        # Set weld to current relative pose
        self._set_weld_to_current_pose(
            self.eq_ids[("right", "ball1")],
            self.hand_r_bid,
            self.ball_bids[0]
        )
        
        # Activate weld for right hand to ball1
        self._activate_weld(self.eq_ids[("right", "ball1")], True)
        self.attached = {"right": "ball1", "left": None}
        
        # Run forward dynamics to apply weld constraint
        mujoco.mj_forward(self.model, self.data)
        # Also run a step to ensure constraint is properly applied
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        info = {
            "attached_right": "ball1",
            "attached_left": None,
            "grip_right": 1.0,
            "grip_left": 0.0,
        }
        
        return obs, info
    
    def step(self, action):
        """Execute one environment step"""
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        
        # Split action: 8 motors + 2 grips
        motor_ctrl = action[:8]
        self.grip_right = float(np.clip((action[8] + 1) / 2, 0.0, 1.0))
        self.grip_left = float(np.clip((action[9] + 1) / 2, 0.0, 1.0))
        
        # Apply motor controls
        self.data.ctrl[self.act_ids] = motor_ctrl
        
        # Simulate physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
            # Check for catching: if gripping and touching, attach
            # Access sensor data directly from sensordata array
            touch_vals = self.data.sensordata[self.touch_sids]
            
            # Right hand catch logic (only ball1)
            if self.grip_right > 0.5 and self.attached["right"] is None:
                if touch_vals[0] > 0:
                    self._try_catch("right", "ball1")
            
            # Left hand catch logic (only ball1)
            if self.grip_left > 0.5 and self.attached["left"] is None:
                if touch_vals[1] > 0:
                    self._try_catch("left", "ball1")
            
            # Release logic: if grip not pressed, release
            # if self.grip_right <= 0.5:
            #     self._release("right")
            # if self.grip_left <= 0.5:
            #     self._release("left")
        
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = False
        
        info = {
            "attached_right": self.attached["right"],
            "attached_left": self.attached["left"],
            "grip_right": self.grip_right,
            "grip_left": self.grip_left,
        }
        
        return obs, float(reward), bool(terminated), bool(truncated), info
    
    def _compute_reward(self):
        """Compute reward based on current state"""
        palm_r = self._site_worldpos("palm_right")
        palm_l = self._site_worldpos("palm_left")
        ball_positions = [self.data.xpos[bid] for bid in self.ball_bids]
        
        # Height reward: encourage balls to stay high
        heights = sum(p[2] for p in ball_positions)
        height_reward = self.reward_weights["height"] * heights
        
        # Distance penalty: discourage balls from being too far from palms
        dist_penalty = sum(
            np.linalg.norm(p - palm_r) + np.linalg.norm(p - palm_l)
            for p in ball_positions
        )
        distance_reward = self.reward_weights["distance"] * dist_penalty
        
        # Touch bonus: reward contact between hands and balls
        touch_vals = self.data.sensordata[self.touch_sids]
        touch_reward = self.reward_weights["touch"] * float(np.sum(touch_vals > 0.0))
        
        # Catch bonus: reward successful catches
        catch_reward = self.reward_weights["catch"] * (
            float(self.attached["right"] is not None) +
            float(self.attached["left"] is not None)
        )
        
        # Survival bonus: small reward for keeping episode alive
        survive_reward = self.reward_weights["survive"]
        
        total_reward = (
            height_reward + distance_reward + touch_reward +
            catch_reward + survive_reward
        )
        
        return total_reward
    
    def _check_termination(self):
        """Check if episode should terminate"""
        ball_positions = [self.data.xpos[bid] for bid in self.ball_bids]
        # Terminate if ball hits the floor (z < 0.06, accounting for floor plane)
        return any(p[2] < 0.06 for p in ball_positions)
    
    def _get_obs(self):
        """Get current observation"""
        obs_list = []
        
        # Base state: joint positions and velocities
        obs_list.append(self.data.qpos.ravel())
        obs_list.append(self.data.qvel.ravel())
        
        # Palm positions and velocities
        palm_r = self._site_worldpos("palm_right")
        palm_l = self._site_worldpos("palm_left")
        vr = self._body_vel(self.hand_r_bid)
        vl = self._body_vel(self.hand_l_bid)
        obs_list.extend([palm_r, vr, palm_l, vl])
        
        # Ball positions and velocities
        for bid in self.ball_bids:
            obs_list.append(self.data.xpos[bid].copy())
            obs_list.append(self._body_vel(bid))
        
        # Touch sensors
        touch_vals = self.data.sensordata[self.touch_sids].copy()
        obs_list.append(touch_vals)
        
        return np.concatenate([x.astype(np.float32) for x in obs_list])
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(
                    self.model, width=self.img_size[0], height=self.img_size[1]
                )
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        elif self.render_mode == "human":
            if self.viewer_sync:
                self.viewer.sync()
            return None
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} not implemented")
    
    def close(self):
        """Clean up resources"""
        if self.renderer is not None:
            self.renderer = None
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Example usage
if __name__ == "__main__":
    import os
    from PIL import Image
    
    # Get the XML path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "assests", "juggler.xml")
    
    # Create environment with higher resolution and no frame skip
    env = JugglerEnv(xml_path=xml_path, render_mode="rgb_array", frame_skip=1, img_size=(1920, 1080))
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Save first frame after reset
    first_frame = env.render()
    if first_frame is not None:
        output_dir = os.path.join(os.path.dirname(current_dir), "render")
        os.makedirs(output_dir, exist_ok=True)
        first_frame_path = os.path.join(output_dir, "first_frame.png")
        Image.fromarray(first_frame).save(first_frame_path)
        print(f"\nFirst frame saved to: {first_frame_path}")
        
        # Print positions for debugging
        hand_r_pos = env.data.xpos[env.hand_r_bid]
        palm_r_pos = env._site_worldpos("palm_right")
        ball1_pos = env.data.xpos[env.ball_bids[0]]
        print(f"Hand right center: {hand_r_pos}")
        print(f"Palm right site: {palm_r_pos}")
        print(f"Ball1 position: {ball1_pos}")
        print(f"Distance hand-ball: {np.linalg.norm(hand_r_pos - ball1_pos):.6f}")
        print(f"Distance palm-ball: {np.linalg.norm(palm_r_pos - ball1_pos):.6f}")
    
    # Store frames for GIF creation
    frames = []
    # Add first frame to frames list
    if first_frame is not None:
        frames.append(Image.fromarray(first_frame))
    
    # Run a few steps
    total_reward = 0.0
    for step in range(100):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render every step and save frame
        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(frame))
        
        if step % 10 == 0:
            print(f"Step {step}: Reward={reward:.3f}, Attached={info}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            obs, info = env.reset()
    
    print(f"Total reward: {total_reward:.3f}")
    
    # Save GIF
    if frames:
        output_dir = os.path.join(os.path.dirname(current_dir), "render")
        os.makedirs(output_dir, exist_ok=True)
        gif_path = os.path.join(output_dir, "test_juggler_env.gif")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,  # milliseconds per frame
            loop=0
        )
        print(f"\nGIF saved to: {gif_path}")
        print(f"Total frames: {len(frames)}")
    
    env.close()

