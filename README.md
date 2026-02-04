# STGCS-3D: Multi-Robot Path Planning in 3D

This repository extends **PBS (Prioritized Bypass Search) + STGCS (Space-Time Graph of Convex Sets)** to support **3D multi-robot path planning**.

## 🚀 Running the 3D Example

**Basic run:**
```bash
python demos/test3d_n2_simple.py
```

**With visualization:**
```bash
python demos/test3d_n2_simple.py --visualize
```

This will:
- Create a 3D environment (10×10×10 cube) with static and dynamic obstacles
- Plan collision-free paths for 2 robots using PBS + STGCS
- Perform collision checking and distance analysis
- Generate 3D visualization frames saved to `output/3d_vis/` (if `--visualize` is used)

## 🔧 Major Changes in Core Code for 3D Support

### 1. **STGCS Dimension Detection** (`mrmp/stgcs.py`)

**Change:** Automatic dimension detection from environment

```python
@staticmethod
def from_env(env, t0=0, tmax=1e2, vlimit=1.0, dt=1e-6) -> STGCS:
    sets = [make_hpolytope(V) for V in env.C_Space]
    dim = 1 + sets[0].ambient_dimension()  # ← KEY CHANGE
    
    stgcs = STGCS(dim, t0=t0, tmax=tmax, vlimit=vlimit, dt=dt)
    # ...
```

**Impact:**
- **2D environment**: `ambient_dimension() = 2` → `dim = 3` (x, y, t)
- **3D environment**: `ambient_dimension() = 3` → `dim = 4` (x, y, z, t)
- Creates **4D space-time graph** for 3D planning

**Why:** STGCS needs to know the full space-time dimensionality to build the graph correctly.

---

### 2. **Collision Checking Dispatcher** (`mrmp/pbs.py:167-178`)

**Change:** Added automatic routing to dimension-specific collision checking functions

```python
def collision_checking(pi_a, pi_b, robot_radius, tmin, tmax) -> bool:
    if pi_a[0].shape[0] == 4:
        return collision_checking_1d(pi_a, pi_b, robot_radius, tmin, tmax)
    elif pi_a[0].shape[0] == 6:
        return collision_checking_2d(pi_a, pi_b, robot_radius, tmin, tmax)
    elif pi_a[0].shape[0] == 8:  # ← NEW: 3D support
        return collision_checking_3d(pi_a, pi_b, robot_radius, tmin, tmax)
    else:
        raise ValueError(f"collision_checking: unsupported dimension {pi_a[0].shape[0]}")
```

**Impact:**
- Automatically detects waypoint format (4/6/8 elements)
- Routes to appropriate collision checking function
- No manual dimension specification needed

**Waypoint Formats:**
- **1D**: `[x0, t0, x1, t1]` (4 elements)
- **2D**: `[x0, y0, t0, x1, y1, t1]` (6 elements)
- **3D**: `[x0, y0, z0, t0, x1, y1, z1, t1]` (8 elements)

---

### 3. **3D Collision Checking Function** (`mrmp/pbs.py:221-238`)

**Change:** New function specifically for 3D collision detection

```python
def collision_checking_3d(pi_a, pi_b, robot_radius, tmin, tmax) -> bool:
    """3D collision checking: waypoint format [x0, y0, z0, t0, x1, y1, z1, t1]"""
    # Extract 3D spatial coordinates (x, y, z) + time
    _a_t0 = [np.concatenate([pi_a[0][0:3], [tmin], pi_a[0][:4]])]  # [x,y,z,tmin,x,y,z,t]
    _a_tf = [np.concatenate([pi_a[-1][-4:], pi_a[-1][-4:-1], [tmax]])]
    # ... similar for pi_b
    
    for xyz_a, xyz_b in product(_a_t0 + pi_a + _a_tf, _b_t0 + pi_b + _b_tf):
        xa, ya = xyz_a[:4], xyz_a[4:]  # [x0,y0,z0,t0] and [x1,y1,z1,t1]
        xb, yb = xyz_b[:4], xyz_b[4:]
        val = min_dist_squared(xa, ya, xb, yb) - (2 * robot_radius)**2
        if not np.allclose(val, 0) and val < 0:
            return True
    
    return False
```

**Key Differences from 2D:**

| Aspect | 2D (`collision_checking_2d`) | 3D (`collision_checking_3d`) |
|--------|------------------------------|------------------------------|
| **Array Slicing** | `pi_a[0][0:2]` (x, y) | `pi_a[0][0:3]` (x, y, z) |
| **Waypoint Extract** | `xy_a[:3], xy_a[3:]` | `xyz_a[:4], xyz_a[4:]` |
| **Spatial Dims** | 2 (x, y) | 3 (x, y, z) |
| **Distance Calc** | 2D Euclidean | 3D Euclidean |

**Why:** The same `min_dist_squared()` algorithm works for both, but array indexing must account for the extra z-coordinate.

---

### 4. **STGCS Constraint Construction** (`mrmp/stgcs.py:37-58`)

**Change:** Velocity constraints automatically adapt to dimension

```python
def __init__(self, dim, order=0, t0=0, tmax=1e2, vlimit=1.0, dt=1e-6):
    self.dim = dim  # 3 for 2D, 4 for 3D
    
    # Velocity constraints use (dim-1) for spatial dimensions
    A_vmax = np.hstack([
         np.eye(self.dim-1),  vlimit * np.ones((self.dim-1, 1)),  # ← Adapts to dim
        -np.eye(self.dim-1), -vlimit * np.ones((self.dim-1, 1))
    ])
    # Time constraint: last dimension is always time
    A_dt = np.array([0] * (self.dim-1) + [1] + [0] * (self.dim-1) + [-1])
```

**Impact:**
- **2D**: `self.dim-1 = 2` → constraints on (x, y) velocities
- **3D**: `self.dim-1 = 3` → constraints on (x, y, z) velocities
- Time dimension is always at index `dim-1` (last position)

**Why:** Velocity limits apply to spatial dimensions only, not time. The code automatically handles 2D vs 3D by using `dim-1`.

---

### 5. **Space Bounds Extraction** (`mrmp/stgcs.py:112-123`)

**Change:** Space bounds extraction adapts to spatial dimension

```python
def try_add_vertex(self, hpoly, itvl, tol=1e-6) -> Optional[Vertex]:
    space_bounds = []
    dims = [int(_i) for _i in range(self.dim - 1)]  # ← Excludes time dimension
    for lb, ub in zip(*get_hpoly_bounds(hpoly, dim=dims)):
        if ub - lb <= tol:
            return
        space_bounds.append(Interval(lb, ub))
    return self.add_vertex(hpoly, itvl, space_bounds)
```

**Impact:**
- **2D**: `range(2)` → extracts bounds for x, y
- **3D**: `range(3)` → extracts bounds for x, y, z
- Time bounds handled separately via `itvl` parameter

**Why:** Space-time vertices need separate bounds for spatial dimensions and time interval.

---

### 6. **Space Division for Obstacle Reservation** (`mrmp/ecd.py:114-160, 281-339`)

**Change:** New 3D space partitioning function for reserving space around moving obstacles/agents

**2D Space Division** (`parallelepiped_side_halfspace_2d`):
```python
def parallelepiped_side_halfspace_2d(xp, xq, halfsize):
    """Generate 2 halfspaces for 2D space (x, y) in 3D space-time (x, y, t)"""
    # Creates 2 halfspaces: left/right around moving line segment
    # Returns 2 HPolyhedron objects
```

**3D Space Division** (`parallelepiped_side_halfspace_3d`):
```python
def parallelepiped_side_halfspace_3d(xp, xq, halfsize):
    """Generate 6 halfspaces for 3D space (x, y, z) in 4D space-time (x, y, z, t)"""
    # Creates 6 halfspaces: top, bottom, left, right, front, back
    # Partitions space around moving cube in 3D
    # Returns 6 HPolyhedron objects
```

**Impact:**
- **2D**: 2 halfspaces partition space around a moving line segment
- **3D**: 6 halfspaces partition space around a moving cube
- Used when reserving space-time regions for higher-priority agents

**Why:** When PBS reserves space for an agent's trajectory, it needs to partition the space-time graph. In 3D, a moving robot occupies a cube, requiring 6 faces (vs 2 edges in 2D) to partition the surrounding space.

**Routing Logic** (`mrmp/ecd.py:120-128`):
```python
dim = stgcs.dim
if dim == 2:
    halfspace_func = parallelepiped_side_halfspace_1d
elif dim == 3:
    halfspace_func = parallelepiped_side_halfspace_2d
elif dim == 4:  # ← NEW: 3D support
    halfspace_func = parallelepiped_side_halfspace_3d
```

**Space Slicing** (`mrmp/ecd.py:163-195`):
- The `slice()` function uses these halfspaces to cut space-time regions
- In 3D, regions are sliced by 6 halfspaces instead of 2
- Creates more refined partitions for better obstacle avoidance

---

## 🎨 Visualizing Space Division: `split_region.py`

The `split_region.py` script demonstrates how 3D space is partitioned when a moving obstacle (represented as a cube) travels through the environment. This visualization illustrates the core concept behind **Exclusion Constraint Decomposition (ECD)** used in STGCS for reserving space-time regions. When a robot or obstacle moves along a trajectory, the surrounding space is divided into **6 regions**: top, bottom, left, right, front, and back. Each frame shows the moving cube (orange) and the 6 partitioned regions (colored boxes) that represent how space is split to avoid collisions. This partitioning mechanism is what allows STGCS to create refined convex sets for collision-free path planning. Run `python split_region.py` to generate animation frames showing this space division process.

---

## 📊 Summary of Core Changes

| Component | 2D Support | 3D Support | Change Type |
|-----------|------------|------------|-------------|
| **STGCS Graph** | 3D (x, y, t) | 4D (x, y, z, t) | Automatic dimension detection |
| **Collision Check** | `collision_checking_2d()` | `collision_checking_3d()` | New function + dispatcher |
| **Waypoint Format** | 6 elements | 8 elements | Array slicing changes |
| **Velocity Constraints** | 2 spatial dims | 3 spatial dims | Uses `dim-1` automatically |
| **Space Bounds** | x, y bounds | x, y, z bounds | Extracts `dim-1` dimensions |
| **Space Division** | 2 halfspaces (left/right) | 6 halfspaces (top/bottom/left/right/front/back) | New `parallelepiped_side_halfspace_3d()` |

## 🎯 Key Design Principles

1. **Automatic Dimension Detection**: No manual configuration needed - dimension inferred from environment
2. **Polymorphic Collision Checking**: Dispatcher routes to correct function based on waypoint size
3. **Dimension-Agnostic Core**: Same algorithms work for 2D and 3D, with dimension-specific details handled automatically
4. **Backward Compatible**: 2D code continues to work without changes

## 📁 Example Output

```
✓ Solution found in 8.45 seconds
  Sum of Costs: 18.23
  Makespan: 12.50

COLLISION CHECK AND DISTANCE ANALYSIS
==========================================
✓ No collisions detected!

Agent-to-Agent Minimum Distance:
  Agent 1 <-> Agent 2: 0.4023 m (at t=6.25s)
```

Frames saved to `output/3d_vis/` (if `--visualize` used)
