#  Reinforcement Learning Agent in Waymax

This project implements a reinforcement learning (RL) agent that learns to drive safely in a simplified traffic scenario using the Waymax simulator.

---

## Project Objective

Train an autonomous driving agent using **Proximal Policy Optimization (PPO)** to:

- Navigate through traffic safely
- Avoid collisions with other vehicles
- Learn smart driving behavior over time

---

##  Project Structure

| File | Description |
|------|-------------|
| `waymax_env.py` | Custom Gymnasium-compatible RL environment simulating multi-agent traffic |
| `train.py`      | Trains the PPO agent using Stable-Baselines3 |
| `eval.py`       | Evaluates the trained model over multiple episodes |
| `ppo_waymax_model.zip` | Saved model weights after training |
| `README.md`     | Project explanation and setup |
| `requirements.txt` | Dependencies list (optional) |

---

##  Environment Details

- **Observation Space**:  
  - Ego vehicle state (x, y, heading, speed)  
  - Relative state of nearby agents (x, y, speed, heading)

- **Action Space**: Discrete (5 actions)  
  - `0`: Accelerate  
  - `1`: Decelerate  
  - `2`: Maintain speed  
  - `3`: Hard brake  
  - `4`: Stop

- **Rewards**:  
  - +1 per safe step  
  - -10 for collision  
  - +20 for completing the episode safely

---

##  Setup Instructions

### 1. Clone & Install Waymax

```bash
git clone https://github.com/waymo-research/waymax.git
cd waymax
pip install -e .
