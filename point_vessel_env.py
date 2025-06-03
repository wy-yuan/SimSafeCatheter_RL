# point_vessel_env.py --------------------------------------------------------
"""
Simple 2-D navigation task for RL:
  • Vessel: poly-line center-line buffered by constant radius.
  • Agent : point-mass (circle radius r_obj).
  • Action: (dx, dy) ∈ [-1,1]²  → position += v_max · action.
  • Observation: relative position to goal, last action.
  • Reward: +5·Δ(progress) -0.01 per step, -1 on collision, +100 on success.
Gymnasium ≥ 0.29  |  Python 3.9+
"""
from __future__ import annotations
from typing import Tuple, Optional, Dict

import math, random, pygame, numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------- geometry helpers -----------------------------------------
def dist_point_to_segment(pt: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance from point `pt` to line-segment p1-p2."""
    v, w = p1, p2
    l2 = np.dot(w - v, w - v)
    if l2 == 0:
        return float(np.linalg.norm(pt - v))
    t = np.clip(np.dot(pt - v, w - v) / l2, 0.0, 1.0)
    proj = v + t * (w - v)
    return float(np.linalg.norm(pt - proj))

def distance_to_polyline(pt: np.ndarray, verts: np.ndarray) -> float:
    """Minimum distance from point to any segment in the poly-line."""
    dists = [dist_point_to_segment(pt, verts[i], verts[i + 1])
             for i in range(len(verts) - 1)]
    return min(dists)

# ---------------------------------------------------------------------------
class PointVesselEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # vessel parameters
    RADIUS   = 1.0        # half-width of vessel (mm or px)
    OBJ_R    = 1.0         # agent radius
    V_MAX    = 1.0         # mm per step at |a|=1
    GOAL_TOL = 2.0         # success within 2 mm

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        # poly-line centerline (mm)
        self.verts = np.array([(0, 0), (30, 40), (15, 90),
                               (-20, 130), (10, 180)], dtype=np.float32)

        # -------- Gym spaces ---------------------------------------------
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(2,), dtype=np.float32)
        # obs: dx, dy, last action (2)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(4,), dtype=np.float32)

        self.pos   = np.zeros(2, np.float32)
        self.goal  = np.zeros(2, np.float32)
        self.steps = 0
        self.max_steps = 10000
        self._last_action = np.zeros(2, np.float32)

        # Pygame viewer
        self._viewer = None
        self.scale = 3  # pixels per mm for rendering

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        # Start near first vertex
        start = self.verts[0] + np.array([-5.0, 5.0])
        self.pos = start.astype(np.float32)
        # Goal near last vertex
        self.goal = self.verts[-2].astype(np.float32)-np.array([10, 10])
        self._last_action[:] = 0
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1).astype(np.float32)
        # move
        self.pos += action * self.V_MAX
        self.steps += 1

        # distances
        d_prev = np.linalg.norm((self.pos - action * self.V_MAX) - self.goal)
        d_now  = np.linalg.norm(self.pos - self.goal)
        progress = d_prev - d_now
        # print("progress:", d_prev, d_now, progress)
        # collision check
        wall_dist = distance_to_polyline(self.pos, self.verts)
        collided = (wall_dist - self.RADIUS) < self.OBJ_R

        # reward
        reward = 5.0 * progress - 0.05
        if collided:
            reward -= 5.0
        terminated = d_now < self.GOAL_TOL and not collided
        truncated  = self.steps >= self.max_steps or collided
        if terminated:
            reward += 100.0

        obs = self._get_obs()
        info = {"distance": d_now, "collided": collided,
                "success": terminated}
        self._last_action = action.copy()
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        dxdy = (self.goal - self.pos) / 200.0        # scale to ≈-1..1
        obs = np.concatenate([dxdy, self._last_action])
        return obs.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self):
        if self.render_mode is None:
            return
        w, h = 500, 500
        if self._viewer is None:
            pygame.init()
            self._viewer = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Point-Vessel RL")
            self._clock = pygame.time.Clock()

        self._viewer.fill((30, 30, 30))
        # draw vessel center‐line
        verts = (self.verts).astype(int)
        pts = [(x * self.scale + w // 2, h - (y * self.scale + 40))
               for x, y in verts]
        pygame.draw.lines(self._viewer, (0, 120, 250), False, pts, 2)
        # # draw walls (simple circles along verts)
        # for p in pts:
        #     pygame.draw.circle(self._viewer, (0, 70, 180), p,
        #                        int(self.RADIUS * self.scale), 1)

        # agent
        ax, ay = (self.pos[0] * self.scale + w // 2,
                  h - (self.pos[1] * self.scale + 40))
        pygame.draw.circle(self._viewer, (0, 255, 0), (int(ax), int(ay)),
                           int(self.OBJ_R * self.scale))

        # goal
        gx, gy = (self.goal[0] * self.scale + w // 2,
                  h - (self.goal[1] * self.scale + 40))
        pygame.draw.circle(self._viewer, (255, 0, 0), (int(gx), int(gy)), 5)

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._viewer is not None:
            pygame.quit()
            self._viewer = None

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = PointVesselEnv(render_mode="human")
    obs, _ = env.reset()
    for _ in range(2000):
        a = env.action_space.sample()  # random policy
        obs, r, term, trunc, info = env.step(a)
        print(r)
        env.render()
        if term or trunc:
            env.reset()
    env.close()
