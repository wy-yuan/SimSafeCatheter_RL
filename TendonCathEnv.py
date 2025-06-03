# catheter_tendon_env_no_twist.py
# ---------------------------------------------------------------------------
# Tendon-actuated catheter (advance + bend only) — no axial rotation.
# Action  : [advance, bend]  ∈ [-1, 1]²
# Obs     : 7-D vector  ->  dx, dy, bend_angle, rail_pos, wall_dist, last_action(2)
# Reward  : shaped distance + smoothness + collision penalty + success bonus
# ---------------------------------------------------------------------------
import math, random, numpy as np
import gymnasium as gym
from gymnasium import spaces
from Box2D.b2 import (world as b2World, chainShape as b2Chain)
from typing import Tuple, Optional, Dict

# -------- constants ---------------------------------------------------------
MM2PX      = 1.0
SEG_LEN_MM = 20
SEG_THICK  = 2
N_SEG      = 5
DEG2RAD    = math.pi / 180

ADV_SPEED  = 50 * MM2PX          # mm per step at |advance|=1
BEND_SPEED = 100 * DEG2RAD       # rad per step at |bend|=1
WALL_THICK = 3
# ---------------------------------------------------------------------------
class TendonCathEnv2DOF(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 vessel_poly: Optional[np.ndarray] = None):
        super().__init__()
        self.viewer = None
        self.render_mode = render_mode
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self._build_vessel(vessel_poly)
        self._build_catheter()

        # --- spaces ---------------------------------------------------------
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float64)

        self.goal   = np.array([120., 120.], np.float64)
        self.max_steps = 1000
        self._last_action = np.zeros(2, np.float64)

    # ------------------------------------------------------------------ #
    #   World construction
    # ------------------------------------------------------------------ #
    def _build_vessel(self, vessel_poly: Optional[np.ndarray]):
        if vessel_poly is None:
            pts = [(0, 0), (20, 40), (10, 80), (-10, 130), (15, 180)]
            vessel_poly = np.array(pts, dtype=np.float64)

        # chain shape
        chain = b2Chain(vertices=[(x * MM2PX, y * MM2PX) for x, y in vessel_poly])
        body = self.world.CreateStaticBody()
        body.CreateFixture(shape=chain, density=0, friction=0)

        # store for distance queries
        self._vessel_vertices = vessel_poly.copy()

    def _build_catheter(self):
        self.segs, self.joints = [], []
        base = self.world.CreateDynamicBody(position=(0, 0), angle=0,
                                            linearDamping=1, angularDamping=1)
        base.CreatePolygonFixture(box=(SEG_LEN_MM / 2 * MM2PX, SEG_THICK / 2 * MM2PX), density=1)
        self.segs.append(base)

        # prismatic rail for insertion
        rail = self.world.CreateStaticBody()
        self.prism = self.world.CreatePrismaticJoint(
            bodyA=rail, bodyB=base,
            anchor=(0, 0), axis=(1, 0),
            lowerTranslation=-10, upperTranslation=260,
            enableLimit=True, maxMotorForce=800, enableMotor=True)

        prev = base
        for i in range(1, N_SEG):
            seg = self.world.CreateDynamicBody(position=(i * SEG_LEN_MM * MM2PX, 0), angle=0,
                                               linearDamping=1, angularDamping=1)
            seg.CreatePolygonFixture(box=(SEG_LEN_MM / 2 * MM2PX, SEG_THICK / 2 * MM2PX), density=1)
            joint = self.world.CreateRevoluteJoint(
                bodyA=prev, bodyB=seg,
                anchor=(i * SEG_LEN_MM * MM2PX, 0),
                lowerAngle=-45 * DEG2RAD, upperAngle=45 * DEG2RAD,
                enableLimit=True, maxMotorTorque=300, enableMotor=True)
            self.segs.append(seg); self.joints.append(joint)
            prev = seg

        # distal third is tendon-driven
        self.tendon_ids = list(range(N_SEG // 3 * 2, N_SEG))

    # ------------------------------------------------------------------ #
    #   Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed); np.random.seed(seed)

        self.goal = np.array([random.uniform(80, 150),
                              random.uniform(120, 190)], np.float64)
        self.steps, self.collided = 0, False
        # reset bodies
        for i, seg in enumerate(self.segs):
            seg.position = (i * SEG_LEN_MM * MM2PX, 0)
            seg.angle = 0
            seg.linearVelocity = (0, 0)
            seg.angularVelocity = 0
        self._last_action.fill(0)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1).astype(np.float64)
        advance, bend = action
        self.prism.motorSpeed = advance * ADV_SPEED
        for jid in self.tendon_ids:
            self.joints[jid-1].motorSpeed = bend * BEND_SPEED

        self.world.Step(1 / 60, 6, 2)
        self.steps += 1

        tip = self._tip_xy()
        dist_prev = np.linalg.norm(tip - self.goal)
        self.world.Step(0, 0, 0)
        tip_now = self._tip_xy()
        dist_now = np.linalg.norm(tip_now - self.goal)

        reward = 10 * (dist_prev - dist_now) - 0.05 * np.linalg.norm(action) - 0.01
        if self._distance_to_wall(tip_now) < SEG_THICK / 2:
            reward -= 5; self.collided = True

        terminated = dist_now < 1.0
        truncated  = self.collided or self.steps >= self.max_steps
        if terminated:
            reward += 100

        obs  = self._get_obs()
        info = {"success": terminated,
                "tip_error_mm": dist_now,
                "collisions": int(self.collided)}
        self._last_action = action.copy()
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    #   Observation helper
    # ------------------------------------------------------------------ #
    def _get_obs(self):
        tip = self._tip_xy()
        dx, dy = (self.goal - tip) / 200.0              # ≈ -1..1
        bend_ang = np.mean([self.joints[j-1].angle for j in self.tendon_ids]) / (45 * DEG2RAD)
        rail_pos = self.prism.translation / 130.0       # normalised
        wall_dist = self._distance_to_wall(tip) / 50.0 - 1.0
        obs = np.concatenate([[dx, dy, bend_ang, rail_pos, wall_dist],
                              self._last_action])
        return obs.astype(np.float64)

    # ------------------------------------------------------------------ #
    #   Geometry helpers
    # ------------------------------------------------------------------ #
    def _tip_xy(self):
        seg = self.segs[-1]
        x, y = seg.transform * (SEG_LEN_MM / 2 * MM2PX, 0)
        return np.array([x, y], np.float64)

    def _distance_to_wall(self, point):
        verts = np.array([(0, 0), (20, 40), (10, 80), (-10, 130), (15, 180)], np.float64)
        return float(np.linalg.norm(verts - point / MM2PX, axis=1).min())

    # --- (optional) rendering same as prior env, omitted for brevity ----
    def render(self):
        if self.render_mode != "human":
            return
        import pygame

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
        for seg in self.segs:
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


if __name__ == "__main__":
    env = TendonCathEnv2DOF(render_mode="human")
    obs, _ = env.reset(seed=0)
    for _ in range(300):
        a = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(a)
        env.render()
        if term or trunc:
            env.reset()
    env.close()