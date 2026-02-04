"""
Simple 3D test with 2 agents - demonstrates 3D support using PBS + STGCS
Note: Planning may take time, but visualization will work once solution is found
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import time
import os, sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from mrmp.pbs import PBS
    from environment.env import Env
    from environment.obstacle import StaticSphere, DynamicSphere
    from mrmp.interval import Interval
except:
    raise ImportError("You should run this script from the root directory")


def create_simple_3d_env():
    """Create a simple 3D environment"""
    # Simple 3D C-Space
    CSpace = [np.array([
        [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [10.0, 10.0, 10.0], [0.0, 10.0, 10.0],
    ])]
    
    # Minimal obstacles
    OStatic = [
        StaticSphere(pos=np.array([5.0, 2.0, 5.0]), radius=0.4),
    ]
    
    ODynamic = [
        DynamicSphere(
            x0=np.array([3.0, 8.0, 4.0]),
            xt=np.array([7.0, 2.0, 6.0]),
            radius=0.6,
            itvl=Interval(0.0, 10.0)
        ),
    ]
    
    return Env("test3d_simple", CSpace, robot_radius=0.2, OStatic=OStatic, ODynamic=ODynamic)


def check_collisions_and_distances(env, sols, dt=0.1):
    """Check for collisions and compute minimum distances between agents and obstacles
    
    Args:
        env: Environment with obstacles
        sols: List of solution trajectories for agents
        dt: Time step for sampling (default 0.1s for fine-grained checking)
    
    Returns:
        dict with collision status and minimum distances
    """
    if not sols:
        print("No solutions to check collisions")
        return None
    
    # Find time range
    tmax = max([sol.itvl.end for sol in sols])
    tmin = min([sol.itvl.start for sol in sols])
    time_steps = np.arange(tmin, tmax + dt/2, dt)
    
    # Initialize tracking variables
    min_agent_agent_dist = float('inf')
    min_agent_agent_time = None
    min_agent_agent_pair = None
    
    min_agent_obs_dists = {}  # {agent_idx: {obs_idx: min_dist}}
    min_agent_obs_times = {}  # {agent_idx: {obs_idx: time}}
    
    collisions_found = []
    
    # Check agent-to-agent collisions and distances
    for i in range(len(sols)):
        for j in range(i + 1, len(sols)):
            sol_i = sols[i]
            sol_j = sols[j]
            
            # Find overlapping time interval
            overlap_start = max(sol_i.itvl.start, sol_j.itvl.start)
            overlap_end = min(sol_i.itvl.end, sol_j.itvl.end)
            
            if overlap_start <= overlap_end:
                # Sample during overlapping time
                for t in time_steps:
                    if overlap_start <= t <= overlap_end:
                        try:
                            pos_i = sol_i.lerp(t)
                            pos_j = sol_j.lerp(t)
                            dist = np.linalg.norm(pos_i - pos_j)
                            # Distance between robot centers minus their radii
                            center_to_center_dist = dist - 2 * env.robot_radius
                            
                            if center_to_center_dist < min_agent_agent_dist:
                                min_agent_agent_dist = center_to_center_dist
                                min_agent_agent_time = t
                                min_agent_agent_pair = (i, j)
                            
                            # Check if collision (distance < 0 means overlap)
                            if center_to_center_dist < 0:
                                collisions_found.append({
                                    'type': 'agent_agent',
                                    'agents': (i, j),
                                    'time': t,
                                    'distance': center_to_center_dist
                                })
                        except:
                            pass
    
    # Check agent-to-obstacle collisions and distances
    for i, sol in enumerate(sols):
        min_agent_obs_dists[i] = {}
        min_agent_obs_times[i] = {}
        
        # Check static obstacles
        for j, obs in enumerate(env.O_Static):
            if isinstance(obs, StaticSphere):
                min_dist = float('inf')
                min_time = None
                
                # Always check at agent's start and end times, plus sampled time steps
                agent_time_steps = set([sol.itvl.start, sol.itvl.end])
                agent_time_steps.update([t for t in time_steps if sol.itvl.start <= t <= sol.itvl.end])
                # Also add some intermediate points for better coverage
                if sol.itvl.end > sol.itvl.start:
                    num_samples = max(10, int((sol.itvl.end - sol.itvl.start) / dt) + 1)
                    agent_time_steps.update(np.linspace(sol.itvl.start, sol.itvl.end, num_samples))
                
                for t in sorted(agent_time_steps):
                    if sol.itvl.start <= t <= sol.itvl.end:
                        try:
                            agent_pos = sol.lerp(t)
                            dist = np.linalg.norm(agent_pos - obs.pos)
                            # Distance between robot center and obstacle center minus their radii
                            center_to_center_dist = dist - env.robot_radius - obs.radius
                            
                            if center_to_center_dist < min_dist:
                                min_dist = center_to_center_dist
                                min_time = t
                            
                            # Check if collision
                            if center_to_center_dist < 0:
                                collisions_found.append({
                                    'type': 'agent_static_obstacle',
                                    'agent': i,
                                    'obstacle': j,
                                    'time': t,
                                    'distance': center_to_center_dist
                                })
                        except Exception as e:
                            # Debug: print error if needed
                            pass
                
                min_agent_obs_dists[i][f'static_{j}'] = min_dist
                min_agent_obs_times[i][f'static_{j}'] = min_time
        
        # Check dynamic obstacles
        for j, obs in enumerate(env.O_Dynamic):
            if isinstance(obs, DynamicSphere):
                min_dist = float('inf')
                min_time = None
                
                # Find overlapping time interval
                overlap_start = max(sol.itvl.start, obs.itvl.start)
                overlap_end = min(sol.itvl.end, obs.itvl.end)
                
                # Build comprehensive set of times to check
                check_times = set()
                # Always include agent start and end times
                check_times.add(sol.itvl.start)
                check_times.add(sol.itvl.end)
                # Check before obstacle movement
                if sol.itvl.start < obs.itvl.start:
                    check_times.update(np.arange(sol.itvl.start, min(obs.itvl.start, sol.itvl.end) + dt/2, dt))
                # Check during overlap
                if overlap_start <= overlap_end:
                    check_times.update(np.arange(overlap_start, overlap_end + dt/2, dt))
                # Check after obstacle movement
                if obs.itvl.end < sol.itvl.end:
                    check_times.update(np.arange(max(obs.itvl.end, sol.itvl.start), sol.itvl.end + dt/2, dt))
                # Add more samples for better coverage
                if sol.itvl.end > sol.itvl.start:
                    num_samples = max(10, int((sol.itvl.end - sol.itvl.start) / dt) + 1)
                    check_times.update(np.linspace(sol.itvl.start, sol.itvl.end, num_samples))
                
                for t in sorted(check_times):
                    if sol.itvl.start <= t <= sol.itvl.end:
                        try:
                            agent_pos = sol.lerp(t)
                            
                            # Get obstacle position at time t
                            if obs.itvl.start <= t <= obs.itvl.end:
                                obs_pos = obs.x(t)
                            elif t < obs.itvl.start:
                                obs_pos = obs.x0
                            else:
                                obs_pos = obs.xt
                            
                            dist = np.linalg.norm(agent_pos - obs_pos)
                            # Distance between robot center and obstacle center minus their radii
                            center_to_center_dist = dist - env.robot_radius - obs.radius
                            
                            if center_to_center_dist < min_dist:
                                min_dist = center_to_center_dist
                                min_time = t
                            
                            # Check if collision
                            if center_to_center_dist < 0:
                                collisions_found.append({
                                    'type': 'agent_dynamic_obstacle',
                                    'agent': i,
                                    'obstacle': j,
                                    'time': t,
                                    'distance': center_to_center_dist
                                })
                        except Exception as e:
                            # Debug: print error if needed
                            pass
                
                min_agent_obs_dists[i][f'dynamic_{j}'] = min_dist
                min_agent_obs_times[i][f'dynamic_{j}'] = min_time
    
    # Check trajectory segments for collisions using env.collision_checking_seg
    for i, sol in enumerate(sols):
        for idx in range(len(sol.trajectory)):
            wp = sol.trajectory[idx]
            if len(wp) >= 8:  # 3D space-time: [x0, y0, z0, t0, x1, y1, z1, t1]
                p = wp[:3]      # Start position
                q = wp[4:7]     # End position
                tp = wp[3]      # Start time
                tq = wp[7]      # End time
                
                # Check collision with obstacles
                if env.collision_checking_seg(p, q, tp, tq):
                    collisions_found.append({
                        'type': 'trajectory_segment_obstacle',
                        'agent': i,
                        'segment': idx,
                        'time_interval': (tp, tq)
                    })
    
    return {
        'collisions': collisions_found,
        'min_agent_agent_dist': min_agent_agent_dist,
        'min_agent_agent_time': min_agent_agent_time,
        'min_agent_agent_pair': min_agent_agent_pair,
        'min_agent_obs_dists': min_agent_obs_dists,
        'min_agent_obs_times': min_agent_obs_times
    }


def print_stgcs_graph_info(env, tmax, vlimit):
    """Print information about STGCS graph division into convex sets"""
    from mrmp.stgcs import STGCS
    
    print("\n" + "=" * 100)
    print("STGCS GRAPH STRUCTURE - CONVEX SET DIVISION")
    print("=" * 100)
    
    # Create initial STGCS graph
    stgcs = STGCS.from_env(env=env, t0=0, tmax=tmax, vlimit=vlimit)
    
    print(f"\nInitial Graph (after environment setup):")
    print(f"  Number of C-Space regions: {len(env.C_Space)}")
    print(f"  Initial convex sets (vertices): {stgcs.G.n_vertices}")
    print(f"  Initial edges (connections): {stgcs.G.n_edges}")
    print(f"  Space-time dimension: {stgcs.dim}D ({stgcs.dim-1} spatial + 1 time)")
    
    # Explain how vertices are created
    print(f"\nHow Convex Sets are Created:")
    print(f"  1. Each C-Space region becomes 1 vertex (time-extruded to space-time)")
    print(f"  2. Dynamic obstacles reserve space, splitting vertices into more regions")
    print(f"  3. Each vertex represents a convex set in {stgcs.dim}D space-time")
    
    # Show C-Space regions
    print(f"\nC-Space Regions:")
    for i, cspace in enumerate(env.C_Space):
        lb = np.min(cspace, axis=0)
        ub = np.max(cspace, axis=0)
        print(f"  Region {i+1}:")
        print(f"    Bounds: [{lb[0]:.1f}, {lb[1]:.1f}, {lb[2]:.1f}] to [{ub[0]:.1f}, {ub[1]:.1f}, {ub[2]:.1f}]")
        print(f"    Volume: {(ub[0]-lb[0]) * (ub[1]-lb[1]) * (ub[2]-lb[2]):.1f} cubic units")
    
    # Show how dynamic obstacles affect the graph
    if env.O_Dynamic:
        print(f"\nDynamic Obstacle Impact:")
        print(f"  Dynamic obstacles will split vertices when reserving space")
        print(f"  This increases the number of convex sets for collision avoidance")
        for j, obs in enumerate(env.O_Dynamic):
            if isinstance(obs, DynamicSphere):
                print(f"  Obstacle {j+1}: Moves from t={obs.itvl.start:.1f}s to t={obs.itvl.end:.1f}s")
                print(f"    This will create additional vertex splits during reservation")
    
    print(f"\nNote: During PBS planning, vertices are further split when agents reserve space")
    print(f"      The final graph may have many more vertices than the initial {stgcs.G.n_vertices}")
    print("=" * 100)


def print_final_stgcs_graph_info(env, sols, tmax, vlimit):
    """Print final STGCS graph state after PBS planning completes"""
    from mrmp.stgcs import STGCS
    from mrmp.ecd import reserve as ecd_reserve
    
    print("\n" + "=" * 100)
    print("FINAL STGCS GRAPH - CONVEX SET DIVISION AFTER PLANNING")
    print("=" * 100)
    
    # Create initial STGCS graph
    stgcs_initial = STGCS.from_env(env=env, t0=0, tmax=tmax, vlimit=vlimit)
    
    # Recreate final graph state by reserving all agent trajectories
    stgcs_final = stgcs_initial.copy()
    for i, sol in enumerate(sols):
        stgcs_final = ecd_reserve(stgcs_final, sol.trajectory, 2 * env.robot_radius)
    
    print(f"\nGraph Statistics:")
    print(f"  Initial convex sets (vertices): {stgcs_initial.G.n_vertices}")
    print(f"  Initial edges: {stgcs_initial.G.n_edges}")
    print(f"  Final convex sets (vertices): {stgcs_final.G.n_vertices}")
    print(f"  Final edges: {stgcs_final.G.n_edges}")
    print(f"  Vertex growth: {stgcs_final.G.n_vertices - stgcs_initial.G.n_vertices} additional vertices")
    print(f"  Edge growth: {stgcs_final.G.n_edges - stgcs_initial.G.n_edges} additional edges")
    print(f"  Growth factor: {stgcs_final.G.n_vertices / stgcs_initial.G.n_vertices:.2f}x")
    
    print(f"\nExplanation:")
    print(f"  - Each vertex represents a convex set in {stgcs_final.dim}D space-time")
    print(f"  - Vertices are split when agents/obstacles reserve space-time regions")
    print(f"  - More splits = finer space division = better collision avoidance")
    print(f"  - Total convex sets after planning: {stgcs_final.G.n_vertices}")
    
    print("=" * 100)


def print_agent_trajectories(sols, dt=0.1):
    """Print detailed trajectory information for all agents"""
    if not sols:
        print("\nNo agent solutions to log")
        return
    
    print("\n" + "=" * 100)
    print("AGENT TRAJECTORIES")
    print("=" * 100)
    print(f"Total agents: {len(sols)}")
    print("=" * 100)
    
    for i, sol in enumerate(sols):
        print(f"\nAgent {i+1}:")
        print("-" * 100)
        
        # Get start and goal positions
        start_pos = sol.lerp(sol.itvl.start)
        goal_pos = sol.lerp(sol.itvl.end)
        
        print(f"  Start Position: [{start_pos[0]:6.2f}, {start_pos[1]:6.2f}, {start_pos[2]:6.2f}]")
        print(f"  Goal Position:  [{goal_pos[0]:6.2f}, {goal_pos[1]:6.2f}, {goal_pos[2]:6.2f}]")
        print(f"  Movement Time: {sol.itvl.start:.2f}s to {sol.itvl.end:.2f}s (duration: {sol.itvl.duration:.2f}s)")
        
        # Calculate distance and average speed
        distance = np.linalg.norm(goal_pos - start_pos)
        avg_speed = distance / sol.itvl.duration if sol.itvl.duration > 0 else 0.0
        print(f"  Distance Traveled: {distance:.3f} m")
        print(f"  Average Speed: {avg_speed:.3f} m/s")
        print(f"  Path Cost: {sol.cost:.3f}")
        
        # Sample trajectory points
        print(f"\n  Trajectory Points (sampled at {dt}s intervals):")
        traj_points = []
        
        # Always include start position
        traj_points.append((sol.itvl.start, start_pos))
        
        # Sample intermediate points
        for t_sample in np.arange(sol.itvl.start + dt, sol.itvl.end, dt):
            if sol.itvl.start < t_sample < sol.itvl.end:
                try:
                    pos = sol.lerp(t_sample)
                    traj_points.append((t_sample, pos))
                except:
                    pass  # Skip if lerp fails
        
        # Always include end position (if different from start)
        if sol.itvl.end > sol.itvl.start:
            traj_points.append((sol.itvl.end, goal_pos))
        
        # Print all trajectory points
        for t, pos in traj_points:
            print(f"    t={t:6.2f}s: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
        
        print(f"  Total trajectory points logged: {len(traj_points)}")
        
        # Also show waypoint segments from sol.trajectory
        if sol.trajectory:
            print(f"\n  Waypoint Segments (from solution):")
            print(f"    Total waypoints: {len(sol.trajectory)}")
            for idx, wp in enumerate(sol.trajectory):
                if len(wp) >= 8:  # 3D space-time: [x0, y0, z0, t0, x1, y1, z1, t1]
                    p1 = wp[:3]      # Start position
                    t1 = wp[3]       # Start time
                    p2 = wp[4:7]     # End position
                    t2 = wp[7]       # End time
                    print(f"    Segment {idx+1}: [{p1[0]:6.2f}, {p1[1]:6.2f}, {p1[2]:6.2f}] @ t={t1:.2f}s -> [{p2[0]:6.2f}, {p2[1]:6.2f}, {p2[2]:6.2f}] @ t={t2:.2f}s")
    
    print("=" * 100)


def print_obstacle_paths(env, dt=0.1):
    """Print detailed path information for moving obstacles"""
    if not env.O_Dynamic:
        print("\nNo dynamic obstacles to log")
        return
    
    print("\n" + "=" * 100)
    print("MOVING OBSTACLE PATHS")
    print("=" * 100)
    print(f"Total dynamic obstacles: {len(env.O_Dynamic)}")
    print("=" * 100)
    
    for j, obs in enumerate(env.O_Dynamic):
        if isinstance(obs, DynamicSphere):
            print(f"\nObstacle {j+1}:")
            print("-" * 100)
            print(f"  Type: DynamicSphere")
            print(f"  Radius: {obs.radius:.3f} m")
            print(f"  Start Position: [{obs.x0[0]:6.2f}, {obs.x0[1]:6.2f}, {obs.x0[2]:6.2f}]")
            print(f"  End Position:   [{obs.xt[0]:6.2f}, {obs.xt[1]:6.2f}, {obs.xt[2]:6.2f}]")
            print(f"  Movement Time: {obs.itvl.start:.2f}s to {obs.itvl.end:.2f}s (duration: {obs.itvl.duration:.2f}s)")
            
            # Calculate velocity
            velocity = obs.velocity
            speed = np.linalg.norm(velocity)
            print(f"  Velocity: [{velocity[0]:6.3f}, {velocity[1]:6.3f}, {velocity[2]:6.3f}] m/s")
            print(f"  Speed: {speed:.3f} m/s")
            
            # Calculate distance traveled
            distance = np.linalg.norm(obs.xt - obs.x0)
            print(f"  Distance Traveled: {distance:.3f} m")
            
            # Sample trajectory points
            print(f"\n  Trajectory Points (sampled at {dt}s intervals):")
            traj_points = []
            
            # Always include start position
            traj_points.append((obs.itvl.start, obs.x0))
            
            # Sample intermediate points
            for t_sample in np.arange(obs.itvl.start + dt, obs.itvl.end, dt):
                if obs.itvl.start < t_sample < obs.itvl.end:
                    pos = obs.x(t_sample)
                    traj_points.append((t_sample, pos))
            
            # Always include end position (if different from start)
            if obs.itvl.end > obs.itvl.start:
                traj_points.append((obs.itvl.end, obs.xt))
            
            # Print all trajectory points
            for t, pos in traj_points:
                print(f"    t={t:6.2f}s: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
            
            print(f"  Total trajectory points logged: {len(traj_points)}")
    
    print("=" * 100)


def print_paths_per_timestep(env, sols, dt=0.5):
    """Print agent and obstacle positions at each timestep"""
    if not sols:
        print("No solutions to print paths")
        return
    
    # Find the maximum time across all agents and obstacles
    tmax = max([sol.itvl.end for sol in sols])
    tmin = min([sol.itvl.start for sol in sols])
    
    # Also consider obstacle movement times
    for obs in env.O_Dynamic:
        if isinstance(obs, DynamicSphere):
            tmax = max(tmax, obs.itvl.end)
            tmin = min(tmin, obs.itvl.start)
    
    # Generate time steps
    time_steps = np.arange(tmin, tmax + dt, dt)
    
    print("\n" + "=" * 100)
    print("AGENT AND OBSTACLE LOCATIONS PER TIMESTEP")
    print("=" * 100)
    print(f"Time step: {dt}s")
    print(f"Time range: {tmin:.1f}s to {tmax:.1f}s")
    print(f"Total agents: {len(sols)}")
    print(f"Static obstacles: {len(env.O_Static)}")
    print(f"Dynamic obstacles: {len(env.O_Dynamic)}")
    print("\n" + "-" * 100)
    print("TARGET LOCATIONS SUMMARY")
    print("-" * 100)
    
    # Log agent start and goal positions
    print("\nAgent Targets:")
    for i, sol in enumerate(sols):
        start_pos = sol.lerp(sol.itvl.start)
        goal_pos = sol.lerp(sol.itvl.end)
        print(f"  Agent {i+1}:")
        print(f"    Start: [{start_pos[0]:6.2f}, {start_pos[1]:6.2f}, {start_pos[2]:6.2f}]")
        print(f"    Goal:  [{goal_pos[0]:6.2f}, {goal_pos[1]:6.2f}, {goal_pos[2]:6.2f}]")
        distance = np.linalg.norm(goal_pos - start_pos)
        duration = sol.itvl.duration
        avg_speed = distance / duration if duration > 0 else 0.0
        print(f"    Distance: {distance:.3f} m, Duration: {duration:.2f} s, Avg Speed: {avg_speed:.3f} m/s")
    
    # Log static obstacle positions (they are targets themselves - fixed positions)
    if env.O_Static:
        print("\nStatic Obstacle Positions (Fixed):")
        for j, obs in enumerate(env.O_Static):
            if isinstance(obs, StaticSphere):
                print(f"  Static Obstacle {j+1}: [{obs.pos[0]:6.2f}, {obs.pos[1]:6.2f}, {obs.pos[2]:6.2f}] (radius: {obs.radius:.2f}m)")
    
    # Log dynamic obstacle start, end positions, and speed
    if env.O_Dynamic:
        print("\nDynamic Obstacle Targets:")
        for j, obs in enumerate(env.O_Dynamic):
            if isinstance(obs, DynamicSphere):
                velocity = obs.velocity
                speed = np.linalg.norm(velocity)
                distance = np.linalg.norm(obs.xt - obs.x0)
                print(f"  Dynamic Obstacle {j+1}:")
                print(f"    Start Position: [{obs.x0[0]:6.2f}, {obs.x0[1]:6.2f}, {obs.x0[2]:6.2f}]")
                print(f"    End Position:   [{obs.xt[0]:6.2f}, {obs.xt[1]:6.2f}, {obs.xt[2]:6.2f}]")
                print(f"    Movement Time: {obs.itvl.start:.2f}s to {obs.itvl.end:.2f}s (duration: {obs.itvl.duration:.2f}s)")
                print(f"    Velocity Vector: [{velocity[0]:6.3f}, {velocity[1]:6.3f}, {velocity[2]:6.3f}] m/s")
                print(f"    Moving Speed: {speed:.3f} m/s")
                print(f"    Distance: {distance:.3f} m")
                print(f"    Radius: {obs.radius:.2f} m")
    
    print("=" * 100)
    
    for t in time_steps:
        print(f"\nTime: {t:.2f}s")
        print("-" * 100)
        
        # Print agent positions
        print("  Agents:")
        for i, sol in enumerate(sols):
            if sol.itvl.start <= t <= sol.itvl.end:
                pos = sol.lerp(t)
                status = "moving"
            elif t < sol.itvl.start:
                pos = sol.lerp(sol.itvl.start)  # Start position
                status = "waiting"
            else:
                pos = sol.lerp(sol.itvl.end)  # Goal position
                status = "arrived"
            
            print(f"    Agent {i+1}: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}] ({status})")
        
        # Print static obstacle positions (always at same location)
        if env.O_Static:
            print("  Static Obstacles:")
            for j, obs in enumerate(env.O_Static):
                if isinstance(obs, StaticSphere):
                    print(f"    Static Obstacle {j+1}: [{obs.pos[0]:6.2f}, {obs.pos[1]:6.2f}, {obs.pos[2]:6.2f}] (radius: {obs.radius:.2f}m)")
        
        # Print dynamic obstacle positions
        if env.O_Dynamic:
            print("  Dynamic Obstacles:")
            for j, obs in enumerate(env.O_Dynamic):
                if isinstance(obs, DynamicSphere):
                    if obs.itvl.start <= t <= obs.itvl.end:
                        obs_pos = obs.x(t)
                        status = "moving"
                    elif t < obs.itvl.start:
                        obs_pos = obs.x0
                        status = "waiting"
                    else:
                        obs_pos = obs.xt
                        status = "stopped"
                    # Calculate current speed if moving
                    speed_info = ""
                    if status == "moving":
                        velocity = obs.velocity
                        speed = np.linalg.norm(velocity)
                        speed_info = f", speed: {speed:.3f} m/s"
                    print(f"    Dynamic Obstacle {j+1}: [{obs_pos[0]:6.2f}, {obs_pos[1]:6.2f}, {obs_pos[2]:6.2f}] ({status}, radius: {obs.radius:.2f}m{speed_info})")
    
    print("=" * 100)


def visualize_3d_simple(env, sols, save_frames=True, dt=0.5, show_interactive=True):
    """3D visualization with frame-by-frame animation saved to files
    
    Args:
        env: Environment with obstacles
        sols: List of solution trajectories for agents
        save_frames: Whether to save frames to disk
        dt: Timestep interval (default 0.5s) - agents and obstacles move at this interval
        show_interactive: Whether to show the final interactive plot (default: True)
    """
    if not sols:
        print("No solutions to visualize")
        return
    
    # Create output directory
    output_dir = "output/3d_vis"
    if save_frames:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving frames to {output_dir}/ (timestep: {dt}s)")
    
    # Find time range
    tmax = max([sol.itvl.end for sol in sols])
    tmin = min([sol.itvl.start for sol in sols])
    # Generate time steps at exactly dt intervals
    time_steps = np.arange(tmin, tmax + dt/2, dt)  # Use dt/2 to handle floating point precision
    
    # Agent colors
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    
    # Pre-compute all agent trajectories for efficiency
    agent_trajectories = []
    for sol in sols:
        traj_points = []
        for wp in sol.trajectory:
            # Trajectory waypoints are in space-time format: [x0, y0, z0, t0, x1, y1, z1, t1] for 3D
            # For 3D: dim=3, so space-time has 4 dimensions, waypoint has 2*4=8 elements
            if len(wp) >= 8:  # 3D space-time: [x0, y0, z0, t0, x1, y1, z1, t1]
                p1 = wp[:3]      # Start position: [x0, y0, z0]
                p2 = wp[4:7]     # End position: [x1, y1, z1] (skip t0 at index 3)
                traj_points.extend([p1, p2])
            elif len(wp) >= 6:  # 2D space-time: [x0, y0, t0, x1, y1, t1]
                p1 = wp[:2]      # Start position: [x0, y0]
                p2 = wp[3:5]     # End position: [x1, y1] (skip t0 at index 2)
                traj_points.extend([p1, p2])
            elif len(wp) >= 3:
                # Fallback: assume first 3 elements are spatial coordinates
                traj_points.append(wp[:3])
        agent_trajectories.append(np.array(traj_points) if traj_points else None)
    
    # Pre-compute obstacle trajectories for efficiency
    obstacle_trajectories = []
    for obs in env.O_Dynamic:
        if isinstance(obs, DynamicSphere):
            # Sample obstacle trajectory points at dt intervals
            obs_traj_points = []
            for t_sample in np.arange(obs.itvl.start, obs.itvl.end + dt/2, dt):
                t_sample = min(t_sample, obs.itvl.end)  # Clamp to valid range
                if obs.itvl.start <= t_sample <= obs.itvl.end:
                    try:
                        obs_pos = obs.x(t_sample)
                        obs_traj_points.append(obs_pos)
                    except:
                        pass  # Skip if x() fails
            # Add start and end positions
            if obs_traj_points:
                obs_traj_points.insert(0, obs.x0)
                obs_traj_points.append(obs.xt)
            obstacle_trajectories.append(np.array(obs_traj_points) if obs_traj_points else None)
        else:
            obstacle_trajectories.append(None)
    
    # Generate frames
    for frame_idx, t in enumerate(time_steps):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(0, 10)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f'3D Multi-Robot Planning - Time: {t:.2f}s', fontsize=14, fontweight='bold')
        
        # Draw static obstacles (always visible)
        legend_handles = []
        for obs in env.O_Static:
            if isinstance(obs, StaticSphere):
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = obs.pos[0] + obs.radius * np.outer(np.cos(u), np.sin(v))
                y = obs.pos[1] + obs.radius * np.outer(np.sin(u), np.sin(v))
                z = obs.pos[2] + obs.radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, alpha=0.3, color='gray', shade=False)
                if len(legend_handles) == 0:  # Add label only once
                    from matplotlib.patches import Patch
                    legend_handles.append(Patch(facecolor='gray', alpha=0.3, label='Static Obstacle'))
        
        # Draw full trajectories (light lines in background)
        for i, traj in enumerate(agent_trajectories):
            if traj is not None and len(traj) > 0:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                       color=colors[i % len(colors)], linewidth=1, alpha=0.2, linestyle='--')
        
        # Draw full obstacle trajectories (light lines in background)
        for j, obs_traj in enumerate(obstacle_trajectories):
            if obs_traj is not None and len(obs_traj) > 0:
                ax.plot(obs_traj[:, 0], obs_traj[:, 1], obs_traj[:, 2], 
                       color='red', linewidth=1, alpha=0.2, linestyle='--')
        
        # Draw agents at current time
        agent_labels_added = set()
        for i, sol in enumerate(sols):
            if sol.itvl.start <= t <= sol.itvl.end:
                pos = sol.lerp(t)
                status = "moving"
            elif t < sol.itvl.start:
                pos = sol.lerp(sol.itvl.start)
                status = "waiting"
            else:
                pos = sol.lerp(sol.itvl.end)
                status = "arrived"
            
            # Draw agent as a sphere
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            radius = env.robot_radius
            x = pos[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = pos[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = pos[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x, y, z, alpha=0.7, color=colors[i % len(colors)], shade=False)
            
            # Add label using proxy artist for legend
            if i not in agent_labels_added:
                from matplotlib.patches import Patch
                legend_handles.append(Patch(facecolor=colors[i % len(colors)], alpha=0.7, label=f'Agent {i+1} ({status})'))
                agent_labels_added.add(i)
            
            # Draw path traveled up to current time (using same dt=0.5 timestep)
            traj = agent_trajectories[i]
            if traj is not None and len(traj) > 0 and sol.itvl.start <= t:
                # Sample trajectory points up to current time using the same timestep (dt)
                traveled_points = []
                t_end = min(t, sol.itvl.end)
                if t_end > sol.itvl.start:
                    # Use the same dt (0.5s) for sampling traveled path to match frame timestep
                    for t_sample in np.arange(sol.itvl.start, t_end + dt/2, dt):
                        t_sample = min(t_sample, sol.itvl.end)  # Clamp to valid range
                        if sol.itvl.start <= t_sample <= sol.itvl.end:
                            try:
                                p = sol.lerp(t_sample)
                                traveled_points.append(p)
                            except:
                                pass  # Skip if lerp fails
                
                if len(traveled_points) > 1:
                    traveled = np.array(traveled_points)
                    ax.plot(traveled[:, 0], traveled[:, 1], traveled[:, 2], 
                           color=colors[i % len(colors)], linewidth=2.5, alpha=0.8)
        
        # Draw dynamic obstacles at current time
        obs_labels_added = set()
        for j, obs in enumerate(env.O_Dynamic):
            if isinstance(obs, DynamicSphere):
                if obs.itvl.start <= t <= obs.itvl.end:
                    obs_pos = obs.x(t)
                    status = "moving"
                elif t < obs.itvl.start:
                    obs_pos = obs.x0
                    status = "waiting"
                else:
                    obs_pos = obs.xt
                    status = "stopped"
                
                # Draw obstacle as a sphere
                u = np.linspace(0, 2 * np.pi, 15)
                v = np.linspace(0, np.pi, 15)
                x = obs_pos[0] + obs.radius * np.outer(np.cos(u), np.sin(v))
                y = obs_pos[1] + obs.radius * np.outer(np.sin(u), np.sin(v))
                z = obs_pos[2] + obs.radius * np.outer(np.ones(np.size(u)), np.cos(v))
                
                ax.plot_surface(x, y, z, alpha=0.6, color='red', shade=False)
                
                # Draw path traveled by obstacle up to current time
                obs_traj = obstacle_trajectories[j]
                if obs_traj is not None and len(obs_traj) > 0 and obs.itvl.start <= t:
                    # Sample obstacle trajectory points up to current time
                    traveled_obs_points = []
                    t_end = min(t, obs.itvl.end)
                    if t_end > obs.itvl.start:
                        # Use the same dt for sampling traveled path to match frame timestep
                        for t_sample in np.arange(obs.itvl.start, t_end + dt/2, dt):
                            t_sample = min(t_sample, obs.itvl.end)  # Clamp to valid range
                            if obs.itvl.start <= t_sample <= obs.itvl.end:
                                try:
                                    p = obs.x(t_sample)
                                    traveled_obs_points.append(p)
                                except:
                                    pass  # Skip if x() fails
                    
                    if len(traveled_obs_points) > 1:
                        traveled_obs = np.array(traveled_obs_points)
                        ax.plot(traveled_obs[:, 0], traveled_obs[:, 1], traveled_obs[:, 2], 
                               color='red', linewidth=2.5, alpha=0.8)
                
                # Add label using proxy artist for legend
                if j not in obs_labels_added:
                    from matplotlib.patches import Patch
                    legend_handles.append(Patch(facecolor='red', alpha=0.6, label=f'Obstacle {j+1} ({status})'))
                    obs_labels_added.add(j)
        
        # Add legend with proxy artists
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper left', fontsize=9)
        
        # Add text info
        info_text = f"Frame: {frame_idx+1}/{len(time_steps)}\nTime: {t:.2f}s"
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_frames:
            frame_path = f"{output_dir}/frame_{frame_idx:04d}_t{t:.2f}s.png"
            plt.savefig(frame_path, dpi=150, bbox_inches='tight')
            if frame_idx % 10 == 0 or frame_idx == len(time_steps) - 1:
                print(f"  Saved frame {frame_idx+1}/{len(time_steps)}: {frame_path}")
        
        plt.close(fig)
    
    if save_frames:
        print(f"\n✓ All {len(time_steps)} frames saved to {output_dir}/")
        print(f"  You can create a video with:")
        print(f"  ffmpeg -r 2 -i {output_dir}/frame_%04d_t*.png -vcodec libx264 -pix_fmt yuv420p {output_dir}/animation.mp4")
    
    # Also show the final frame interactively
    if show_interactive:
        print("\nShowing final frame interactively...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(0, 10)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title('3D Multi-Robot Planning - Final State', fontsize=14, fontweight='bold')
        
        # Draw everything for final state
        t_final = tmax
        for obs in env.O_Static:
            if isinstance(obs, StaticSphere):
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                x = obs.pos[0] + obs.radius * np.outer(np.cos(u), np.sin(v))
                y = obs.pos[1] + obs.radius * np.outer(np.sin(u), np.sin(v))
                z = obs.pos[2] + obs.radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, alpha=0.3, color='gray', shade=False)
        
        for i, (sol, traj) in enumerate(zip(sols, agent_trajectories)):
            if traj is not None and len(traj) > 0:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                       color=colors[i % len(colors)], linewidth=2, label=f'Agent {i+1}')
                final_pos = sol.lerp(t_final)
                u = np.linspace(0, 2 * np.pi, 15)
                v = np.linspace(0, np.pi, 15)
                radius = env.robot_radius
                x = final_pos[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = final_pos[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = final_pos[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, alpha=0.7, color=colors[i % len(colors)], shade=False)
        
        for j, (obs, obs_traj) in enumerate(zip(env.O_Dynamic, obstacle_trajectories)):
            if isinstance(obs, DynamicSphere):
                # Draw full obstacle trajectory
                if obs_traj is not None and len(obs_traj) > 0:
                    ax.plot(obs_traj[:, 0], obs_traj[:, 1], obs_traj[:, 2], 
                           color='red', linewidth=2, label=f'Obstacle {j+1}')
                obs_pos = obs.xt if t_final >= obs.itvl.end else obs.x(t_final)
                u = np.linspace(0, 2 * np.pi, 15)
                v = np.linspace(0, np.pi, 15)
                x = obs_pos[0] + obs.radius * np.outer(np.cos(u), np.sin(v))
                y = obs_pos[1] + obs.radius * np.outer(np.sin(u), np.sin(v))
                z = obs_pos[2] + obs.radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, alpha=0.6, color='red', shade=False)
        
        ax.legend()
        plt.show()


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='3D Multi-Robot Planning Test')
    parser.add_argument('--visualize', action='store_true', 
                       help='Enable visualization (default: False)')
    args = parser.parse_args()
    
    env = create_simple_3d_env()
    
    starts = [np.array([1.0, 1.0, 1.0]), np.array([9.0, 9.0, 9.0])]
    goals = [np.array([9.0, 9.0, 9.0]), np.array([1.0, 1.0, 1.0])]
    T0s = [0.0, 0.0]
    
    print("=" * 60)
    print("3D Multi-Robot Planning Test")
    print("=" * 60)
    print(f"Environment: {env.name}")
    print(f"Dimension: {env.dim}D space")
    print(f"Robot radius: {env.robot_radius}")
    print(f"Static obstacles: {len(env.O_Static)}")
    print(f"Dynamic obstacles: {len(env.O_Dynamic)}")
    print(f"\nAgent 1: {starts[0]} -> {goals[0]}")
    print(f"Agent 2: {starts[1]} -> {goals[1]}")
    print("\nStarting PBS planning (this may take a while for 3D)...")
    print("=" * 60)
    
    # Log STGCS graph structure information
    print_stgcs_graph_info(env, tmax=25.0, vlimit=2.0)
    
    ts = time.perf_counter()
    sol, _ = PBS(
        env, tmax=25.0, vlimit=2.0, 
        starts=starts, goals=goals, t0s=T0s, 
        timeout_secs=600, scaler_multiplier=3
    )
    elapsed = time.perf_counter() - ts
    
    if sol:
        # Log final STGCS graph state after planning
        print_final_stgcs_graph_info(env, sol, tmax=25.0, vlimit=2.0)
        print(f"\n✓ Solution found in {elapsed:.2f} seconds")
        print(f"  Sum of Costs: {sum([p.cost for p in sol]):.2f}")
        print(f"  Makespan: {max([p.itvl.end for p in sol]):.2f}")
        
        # Collision checking and distance analysis
        print("\n" + "=" * 100)
        print("COLLISION CHECK AND DISTANCE ANALYSIS")
        print("=" * 100)
        collision_results = check_collisions_and_distances(env, sol, dt=0.1)
        
        if collision_results:
            # Print agent-to-agent distances
            if collision_results['min_agent_agent_pair'] is not None:
                i, j = collision_results['min_agent_agent_pair']
                dist = collision_results['min_agent_agent_dist']
                t = collision_results['min_agent_agent_time']
                print(f"\nAgent-to-Agent Minimum Distance:")
                print(f"  Agent {i+1} <-> Agent {j+1}: {dist:.4f} m (at t={t:.2f}s)")
                if dist < 0:
                    print(f"  ⚠️  COLLISION DETECTED! Overlap of {abs(dist):.4f} m")
                elif dist < 0.1:
                    print(f"  ⚠️  WARNING: Very close! Distance < 0.1m")
            else:
                print(f"\nAgent-to-Agent Minimum Distance: N/A (no overlapping time)")
            
            # Print agent-to-obstacle distances
            print(f"\nAgent-to-Obstacle Minimum Distances:")
            for agent_idx, obs_dists in collision_results['min_agent_obs_dists'].items():
                print(f"\n  Agent {agent_idx+1}:")
                if not obs_dists:
                    print(f"    (No obstacles to check)")
                else:
                    for obs_key, min_dist in obs_dists.items():
                        min_time = collision_results['min_agent_obs_times'][agent_idx][obs_key]
                        if min_time is not None:
                            if obs_key.startswith('static_'):
                                obs_idx = int(obs_key.split('_')[1])
                                print(f"    Static Obstacle {obs_idx+1}: {min_dist:.4f} m (at t={min_time:.2f}s)")
                                if min_dist < 0:
                                    print(f"      ⚠️  COLLISION DETECTED! Overlap of {abs(min_dist):.4f} m")
                                elif min_dist < 0.1:
                                    print(f"      ⚠️  WARNING: Very close! Distance < 0.1m")
                            elif obs_key.startswith('dynamic_'):
                                obs_idx = int(obs_key.split('_')[1])
                                print(f"    Dynamic Obstacle {obs_idx+1}: {min_dist:.4f} m (at t={min_time:.2f}s)")
                                if min_dist < 0:
                                    print(f"      ⚠️  COLLISION DETECTED! Overlap of {abs(min_dist):.4f} m")
                                elif min_dist < 0.1:
                                    print(f"      ⚠️  WARNING: Very close! Distance < 0.1m")
                        else:
                            # min_time is None means agent never encountered this obstacle during its trajectory
                            if obs_key.startswith('static_'):
                                obs_idx = int(obs_key.split('_')[1])
                                if min_dist == float('inf'):
                                    print(f"    Static Obstacle {obs_idx+1}: Not encountered (agent trajectory doesn't overlap)")
                                else:
                                    print(f"    Static Obstacle {obs_idx+1}: {min_dist:.4f} m (time not recorded)")
                            elif obs_key.startswith('dynamic_'):
                                obs_idx = int(obs_key.split('_')[1])
                                if min_dist == float('inf'):
                                    print(f"    Dynamic Obstacle {obs_idx+1}: Not encountered (agent trajectory doesn't overlap)")
                                else:
                                    print(f"    Dynamic Obstacle {obs_idx+1}: {min_dist:.4f} m (time not recorded)")
            
            # Print collision summary
            if collision_results['collisions']:
                print(f"\n⚠️  COLLISION SUMMARY:")
                print(f"  Total collisions detected: {len(collision_results['collisions'])}")
                for coll in collision_results['collisions']:
                    if coll['type'] == 'agent_agent':
                        print(f"    - Agent {coll['agents'][0]+1} <-> Agent {coll['agents'][1]+1} at t={coll['time']:.2f}s (overlap: {abs(coll['distance']):.4f}m)")
                    elif coll['type'] == 'agent_static_obstacle':
                        print(f"    - Agent {coll['agent']+1} <-> Static Obstacle {coll['obstacle']+1} at t={coll['time']:.2f}s (overlap: {abs(coll['distance']):.4f}m)")
                    elif coll['type'] == 'agent_dynamic_obstacle':
                        print(f"    - Agent {coll['agent']+1} <-> Dynamic Obstacle {coll['obstacle']+1} at t={coll['time']:.2f}s (overlap: {abs(coll['distance']):.4f}m)")
                    elif coll['type'] == 'trajectory_segment_obstacle':
                        print(f"    - Agent {coll['agent']+1} trajectory segment {coll['segment']} collides with obstacle (time: {coll['time_interval'][0]:.2f}s - {coll['time_interval'][1]:.2f}s)")
            else:
                print(f"\n✓ No collisions detected!")
        
        print("=" * 100)
        
        # Print detailed agent trajectories - always log agent trajectories
        print_agent_trajectories(sol, dt=0.1)
        
        # Print detailed obstacle paths - always log obstacle trajectories
        print_obstacle_paths(env, dt=0.1)
        
        # Print paths per timestep (0.5s intervals) - always log trajectory
        dt = 0.5
        print_paths_per_timestep(env, sol, dt=dt)
        
        # Optional: Visualization
        if args.visualize:
            print(f"\nGenerating visualization frames at {dt}s timestep intervals...")
            visualize_3d_simple(env, sol, save_frames=True, dt=dt, show_interactive=True)
        else:
            print("\n(Skipping visualization. Use --visualize to enable.)")
    else:
        print(f"\n✗ No solution found after {elapsed:.2f} seconds")
        print("This may be due to:")
        print("  - Obstacles blocking all paths")
        print("  - Timeout too short")
        print("  - Need to adjust start/goal positions")

