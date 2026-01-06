# ✈️ Gate Assignment Optimization (GAP)

This repository implements a **Mixed-Integer Programming (MIP)** model for solving an airport gate assignment problem using operational flight data and historic gate usage patterns.

The project is designed to be:
- cleanly structured  
- easy to reproduce  
- suitable for GitHub portfolios

---

## 📁 Repository Structure

```
GAP/
|
|-- gap_main.py                # Main optimization model (MIP)
|-- historic_analysis.py       # Historic data analysis & plots
|-- optimized_analysis.py      # Optimized solution analysis & plots
|
|-- flight_data.xlsx           # Input data (one-day + historic)
|
|-- csv_outputs/               # Generated CSV outputs
|-- plot_outputs/              # Generated plots
|
|-- requirements.txt
|-- README.md
```

---

## 📊 Input Data

The model reads data from `flight_data.xlsx` containing two sheets.

### 1️⃣ one_day_departures (Optimization Input)

Required columns:
- **flight_code** – Unique flight identifier  
- **boardingtime** – Boarding start time (HH:MM)  
- **departure_time** – Scheduled departure time (HH:MM)  
- **passengers** – Passenger count  
- **aircraft_wing_span** – Aircraft wingspan (meters)  

### 2️⃣ historic_gate_allocation (Historic Reference)

Used to derive gate capacity distributions and benchmark performance.

Required columns:
- **departure_gate** – Gate / stand number  
- **passengers** – Passenger count  
- **boarding_time** or **flight_time** – Time reference  
- **coaches** (optional) – Number of coaches used  

---

## 🧮 Optimization Model

### Sets
- **F** : set of flights  
- **G** : set of gates  
- **T** : set of discretized time slots  
- **C = {0,1}** : assignment modes  
  - 0 = contact gate  
  - 1 = coached (remote stand)  

### Parameters
- **P_f** : passengers of flight *f*  
- **B_f** : boarding start time of flight *f* (minutes)  
- **D_f** : departure time of flight *f* (minutes)  
- **A_{f,t}** : 1 if flight *f* occupies time slot *t*  
- **K_g** : passenger capacity of gate *g* (derived from historic data)  
- **w_h(f)** : hour-of-day weight for flight *f*  

### Decision Variables
- **x_{f,g,c} ∈ {0,1}**  
  = 1 if flight *f* is assigned to gate *g* using mode *c*  

- **z_g ≥ 0**  
  = total passengers assigned to gate *g*  

- **z_max, z_min ≥ 0**  
  = maximum and minimum gate passenger loads  

---

## 🎯 Objective Function

Minimize coached flights while balancing passenger load across gates:

```
minimize
    Σ_f Σ_g w_h(f) · x_f,g,1
    +
    λ · (z_max − z_min) / (Σ_f P_f)
```

- First term penalizes coached (remote) assignments, especially during peak hours  
- Second term minimizes passenger load imbalance  

---

## 📐 Constraints

### 1️⃣ Flight Assignment
Each flight must be assigned exactly once:

```
Σ_g Σ_c x_f,g,c = 1     ∀ f ∈ F
```

### 2️⃣ Gate Time Compatibility
At most one flight may occupy a gate at any time slot:

```
Σ_f A_f,t · Σ_c x_f,g,c ≤ 1     ∀ g ∈ G, t ∈ T
```

### 3️⃣ Gate Passenger Capacity

```
z_g = Σ_f P_f · Σ_c x_f,g,c     ∀ g ∈ G
z_g ≤ K_g                      ∀ g ∈ G
```

### 4️⃣ Load Spread Definition

```
z_g ≤ z_max
z_g ≥ z_min     ∀ g ∈ G
```

---

## 📈 Outputs

### CSV Outputs
- `solution_gap.csv` – Optimized flight-to-gate assignments  
- `gate_capacity.csv` – Computed gate capacities  
- Historic KPI tables  
- Optimized KPI tables  

### Plots
- Historic passenger distribution by gate  
- Historic coach usage and PSL by hour  
- Optimized gate timelines  
- Optimized passenger load distribution  

---

## ▶️ How to Run

### 1️⃣ Activate environment
```bash
conda activate pythonProject
```

### 2️⃣ Run optimization
```bash
python gap_main.py
```

### 3️⃣ Optional: Run analysis separately
```python
import historic_analysis as ha
ha.run_historic_analysis(hist_df)

import optimized_analysis as oa
oa.run_optimized_analysis(solution_df)
```

---

## 📦 Requirements

`requirements.txt`:

```
numpy
pandas
matplotlib
openpyxl
gurobipy
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🧠 Notes

- The model is intentionally kept simple and transparent.  
- Additional constraints (wingspan–gate compatibility, remote-only gates, buffer tuning)
  can be modularly added.  
- Designed for technical assessments, interviews, and portfolio demonstration.  

---

## 👤 Author

**Huseyin Ugur Yildiz**
