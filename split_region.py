import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

# Parameters
space_size = 10
robot_radius = 0.5
p_start = np.array([2, 2, 2])
p_end   = np.array([8, 8, 6])
num_steps = 10

positions = np.linspace(p_start, p_end, num_steps)
times = np.linspace(5, 8, num_steps)

# Helper to draw a box given x,y,z ranges
def draw_box(ax, xrange, yrange, zrange, color, alpha=0.15):
    x0, x1 = xrange
    y0, y1 = yrange
    z0, z1 = zrange
    verts = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                      [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    faces = [
        [verts[j] for j in [0,1,2,3]],  # bottom
        [verts[j] for j in [4,5,6,7]],  # top
        [verts[j] for j in [0,1,5,4]],  # front
        [verts[j] for j in [2,3,7,6]],  # back
        [verts[j] for j in [0,3,7,4]],  # left
        [verts[j] for j in [1,2,6,5]],  # right
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, alpha=alpha, edgecolors='k', linewidths=0.3))

# Set up output folder for frames
os.makedirs("frames", exist_ok=True)

# Generate animation frames
for step, (pos, t) in enumerate(zip(positions, times)):
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_zlim(0, space_size)
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Environment cube
    draw_box(ax, (0,10), (0,10), (0,10), color='lightgray', alpha=0.05)

    # Current "tube" center cube
    cx, cy, cz = pos
    draw_box(ax, (cx-robot_radius, cx+robot_radius),
                  (cy-robot_radius, cy+robot_radius),
                  (cz-robot_radius, cz+robot_radius),
                  color='orange', alpha=0.6)

    # Compute 6 partition boxes
    # top, bottom, left, right, front, back relative to the moving cube
    x0, x1 = cx-robot_radius, cx+robot_radius
    y0, y1 = cy-robot_radius, cy+robot_radius
    z0, z1 = cz-robot_radius, cz+robot_radius

    # Environment boundaries
    X0, X1, Y0, Y1, Z0, Z1 = 0,10,0,10,0,10

    # 6 boxes (space cut by cube)
    draw_box(ax, (X0,X1), (Y0,Y1), (z1,Z1), color='blue', alpha=0.15)    # top
    draw_box(ax, (X0,X1), (Y0,Y1), (Z0,z0), color='cyan', alpha=0.15)    # bottom
    draw_box(ax, (X0,x0), (Y0,Y1), (z0,z1), color='green', alpha=0.15)   # left
    draw_box(ax, (x1,X1), (Y0,Y1), (z0,z1), color='lime', alpha=0.15)    # right
    draw_box(ax, (X0,X1), (Y0,y0), (z0,z1), color='red', alpha=0.15)     # front
    draw_box(ax, (X0,X1), (y1,Y1), (z0,z1), color='magenta', alpha=0.15) # back

    ax.text(0,0,10.2, f"t = {t:.1f}s", fontsize=12)

    ax.set_title("3D Space Partitioned by Moving Tube (ECD analogy)")
    plt.tight_layout()
    plt.savefig(f"frames/frame_{step:02d}.png", dpi=200)
    plt.close(fig)

print("Frames saved in ./frames/. You can animate them with:")
print("!ffmpeg -r 2 -i frames/frame_%02d.png -vcodec libx264 -pix_fmt yuv420p stgcs_tube_division.mp4")
