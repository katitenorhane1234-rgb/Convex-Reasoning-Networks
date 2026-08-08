# Convex Reasoning Networks (CRN)

Convex Reasoning Networks (CRN) is a research-oriented neural architecture
based on convex geometry, constrained state dynamics, and metric-aware
proximal updates.

## Core Update Rule

x_{t+1} = Prox_C^M((I + A)x_t + B g_t)

The model combines:

- Convex geometry
- Learned context representations
- Metric-aware proximal operators
- Contractive state dynamics
- Neural state evolution

## Project Structure

```text
crn/
├── config.py
├── utils.py
├── geometry.py
├── solvers.py
├── model.py
├── dataset.py
├── losses.py
├── train.py
├── evaluate.py
├── plots.py
├── ablation.py
├── main.py
├── requirements.txt
└── README.md
