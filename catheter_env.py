# cath_env.py
# ---------------------------------------------------------------------------
# 2-D Steerable-Catheter environment for Gymnasium / Stable-Baselines3.
#
# * World engine:    Box2D (zero gravity)
# * Observation:     tip & goal positions, segment angles, last action,
#                    distance-to-nearest wall    → 12-dim float32 vector
# * Action:          [Δθ_tip, advance]  ∈  [-1,1]²  (scaled to physical max)
# * Reward:          ▸ −Δ(dist) per step
#                    ▸ −|steer|·0.1   (energy / smoothness)
#                    ▸ −5    on collision
#                    ▸ +100  on success (<1 mm)
#
# Author:  you  (c) 2025
# ---------------------------------------------------------------------------
import math, random, os
from pathlib import Path
from typing import Tuple, Optional, Dict

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import Box2D
from Box2D import b2Vec2, b2ChainShape
from Box2D.b2 import (world as b2World, polygonShape as b2Polygon,
                      chainShape as b2Chain, dynamicBody as b2DynamicBody,
                      revoluteJoint as b2RevoluteJoint)

# ------------------------- helpers -----------------------------------------
MM2PX = 1.0          # 1 mm ⇒ 1 Box2D unit (keep scale 1:1 for clarity)
DEG2RAD = math.pi / 180
SEG_LEN_MM = 10
SEG_THICK_MM = 5
N_SEG = 6
ADVANCE_SPEED_MM = 4          # per action step at |adv|=1
STEER_SPEED_DEG = 6           # per action step at |angle|=1
WALL_THICK = 3

# ---------------------------------------------------------------------------
class Cath2DEnv(gym.Env):
    """
    2-D articulated catheter navigating a static vessel lumen.
    Action  : [steer, advance]
    Obs     : 12-D float (normalised)
    Episode : done on success, collision or steps>1000
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 vessel_poly: Optional[np.ndarray] = None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None

        # --- action & observation spaces -----------------------------------
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        obs_hi = np.ones(17, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_hi, obs_hi, dtype=np.float32)

        # --- physics world --------------------------------------------------
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self._build_vessel(vessel_poly)
        self._build_catheter()
        self.goal = np.array([100.0, 100.0])        # placeholder, mm
        self.steps = 0
        self.collided = False

    # ------------------------------------------------------------------ #
    #  Construction helpers
    # ------------------------------------------------------------------ #
    def _build_vessel(self, vessel_poly: Optional[np.ndarray]):
        """
        Build static vessel wall fixtures from a polyline buffered outward.
        If no poly provided, create a simple gently-curved tube.
        """
        if vessel_poly is None:
            pts = [(0, 0), (20, 40), (10, 80), (-5, 120)]
            vessel_poly = np.array(pts, dtype=np.float64)

        # chain shape
        chain = b2Chain(vertices=[(x * MM2PX, y * MM2PX) for x, y in vessel_poly])
        body = self.world.CreateStaticBody()
        body.CreateFixture(shape=chain, density=0, friction=0)

        # store for distance queries
        self._vessel_vertices = vessel_poly.copy()

    def _build_catheter(self):
        """
        Build N_SEG rectangular dynamic bodies connected by revolute joints.
        """
        self.cat_segs = []
        self.cat_joints = []
        base = self.world.CreateDynamicBody(
            position=(0, 0),
            angle=0,
            linearDamping=1,
            angularDamping=1,
        )
        box = base.CreatePolygonFixture(
            box=(SEG_LEN_MM / 2 * MM2PX, SEG_THICK_MM / 2 * MM2PX),
            density=1.0,
            friction=0.4)
        self.cat_segs.append(base)

        # chained segments
        prev = base
        for i in range(1, N_SEG):
            seg = self.world.CreateDynamicBody(
                position=(i * SEG_LEN_MM * MM2PX, 0),
                angle=0,
                linearDamping=1,
                angularDamping=1)
            seg.CreatePolygonFixture(
                box=(SEG_LEN_MM / 2 * MM2PX, SEG_THICK_MM / 2 * MM2PX),
                density=1.0,
                friction=0.4)
            joint = self.world.CreateRevoluteJoint(
                bodyA=prev,
                bodyB=seg,
                anchor=(i * SEG_LEN_MM * MM2PX, 0),
                lowerAngle=-30 * DEG2RAD,
                upperAngle=30 * DEG2RAD,
                enableLimit=True,
                maxMotorTorque=500,
                enableMotor=True,
            )
            self.cat_segs.append(seg)
            self.cat_joints.append(joint)
            prev = seg

    # ------------------------------------------------------------------ #
    #  Gym API
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        # randomise starting tip and goal
        start = np.array([0.0, 0.0])
        goal_candidates = [(20, 120), (-15, 160), (25, 150)]
        self.goal = np.array(random.choice(goal_candidates), dtype=np.float64)
        self.steps, self.collided = 0, False

        # reset body positions
        for i, seg in enumerate(self.cat_segs):
            seg.position = (i * SEG_LEN_MM * MM2PX, 0)
            seg.angle = 0
            seg.linearVelocity = (0, 0)
            seg.angularVelocity = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        steer, advance = action.astype(np.float64)
        steer *= STEER_SPEED_DEG * DEG2RAD
        advance *= ADVANCE_SPEED_MM * MM2PX

        # apply steering torque to tip joint
        self.cat_joints[-1].motorSpeed = steer

        # advance all segments by same linear impulse
        impulse = (advance, 0)
        for seg in self.cat_segs:
            seg.ApplyLinearImpulse(impulse=impulse, point=seg.worldCenter, wake=True)

        # physics step
        self.world.Step(1 / 60, 6, 2)
        self.steps += 1

        # check collision (simplistic broad-phase)
        tip = self._tip_xy()
        if self._distance_to_wall(tip) < SEG_THICK_MM / 2:
            self.collided = True

        # compute reward
        dist_old = np.linalg.norm(tip - self.goal)
        self.world.Step(0, 0, 0)  # recompute broad-phase cache
        tip_new = self._tip_xy()
        dist_new = np.linalg.norm(tip_new - self.goal)
        reward = 10 * (dist_old - dist_new) - 0.1 * abs(steer)
        if self.collided:
            reward -= 5.0

        terminated = dist_new < 1.0
        truncated = self.steps > 1000 or self.collided
        if terminated:
            reward += 100.0

        obs = self._get_obs()
        info = {"success": terminated,
                "collisions": int(self.collided),
                "tip_error_mm": dist_new}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        tip = self._tip_xy()
        dx, dy = (self.goal - tip) / 200  # normalise ≈ -1..1
        angles = np.array([seg.angle for seg in self.cat_segs], dtype=np.float32)
        angles = np.sin(angles).tolist() + np.cos(angles).tolist()
        last_action = np.array([0.0, 0.0])  # placeholder
        wall_dist = self._distance_to_wall(tip) / 50.0 - 1.0  # ~-1..1
        obs = np.concatenate([[dx, dy], angles[:], last_action, [wall_dist]])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Geometry helpers
    # ------------------------------------------------------------------ #
    def _tip_xy(self) -> np.ndarray:
        tip_body = self.cat_segs[-1]
        x, y = tip_body.transform * (SEG_LEN_MM / 2 * MM2PX, 0)
        return np.array([x, y], dtype=np.float32)

    def _distance_to_wall(self, point: np.ndarray) -> float:
        """cheap distance: min Euclidean to discrete vessel verts"""
        dists = np.linalg.norm(self._vessel_vertices - point / MM2PX, axis=1)
        return float(dists.min())

    # ------------------------------------------------------------------ #
    #  Rendering
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode != "human":
            return
        import pygame
        if self.viewer is None:
            pygame.init()
            self.win_size = 600
            self.scale = 2
            self.screen = pygame.display.set_mode((self.win_size, self.win_size))
            self.clock = pygame.time.Clock()
        self.screen.fill((50, 50, 50))

        # draw vessel polyline
        verts = (self._vessel_vertices * self.scale).astype(int)
        pygame.draw.lines(self.screen, (0, 100, 200), False,
                          [(x + self.win_size // 2, self.win_size - (y + 50)) for x, y in verts], WALL_THICK)

        # draw catheter segments
        for seg in self.cat_segs:
            x, y = seg.position * self.scale
            y = self.win_size - (y + 50)
            pygame.draw.circle(self.screen, (0, 255, 0), (int(x + self.win_size // 2), int(y)), 3)

        # goal
        gx, gy = (self.goal * self.scale).astype(int)
        pygame.draw.circle(self.screen, (255, 0, 0),
                           (gx + self.win_size // 2, self.win_size - (gy + 50)), 5, 1)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.viewer:
            import pygame
            pygame.quit()
            self.viewer = None

# ---------------------------------------------------------------------------
# simple smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = Cath2DEnv(render_mode="human")
    obs, _ = env.reset(seed=0)
    for _ in range(300):
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        env.render()
        if term or trunc:
            env.reset()
    env.close()
