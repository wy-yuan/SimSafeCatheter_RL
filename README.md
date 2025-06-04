# SimSafeCatheter RL
Minimal end-to-end project that **learns to steer a 2-D articulated catheter** tip through a virtual vessel using PPO (Stable-Baselines3).

<img src='gif/out.gif'>

---
### 1 Quick install

```bash
# 1-line environment
conda create -n catheter_rl python=3.10 -y
conda activate catheter_rl
# deps
pip install swig
pip install box2d-py
pip install gymnasium[box2d] stable-baselines3[extra] numpy matplotlib rich
```

### 2 Repo layout

| File                      | Purpose                                                                                            |
|---------------------------|----------------------------------------------------------------------------------------------------|
| `point_vessel_env.py`     | Gymnasium environment – one point represent catheter tip, reward collision logic. (To be upgraded) |
| `train.py`                | Trains PPO.                                                                                        |
| `eval.py`                 | Batch-evaluate trained weights; prints success rate, mean tip error, collisions/ep.                |
| `visualize.py`            | Live Pygame viewer; run with or without a model: <br>`python visualize.py ppo_cath_tip.zip`.       |
| `README.md` (*this file*) | Quick-start guide.                                                                                 |

---

### 3 Train

```bash
python train.py              # creates ppo_cath_tip.zip under /models
```

---

### 4  Evaluate

```bash
python eval.py models/ppo_cath_tip.zip
# → Success 97 %, tip err 0.8 mm, collisions 0.1 / ep
```

---

### 5 Visualise a rollout

```bash
python visualize.py models/ppo_cath_tip.zip
```

Green dots = catheter segments • Blue polyline = vessel • Red circle = target.

---

### 6 Key achievements

To be done.

---


