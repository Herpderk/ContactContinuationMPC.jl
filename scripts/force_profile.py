import mujoco
import numpy as np
import matplotlib.pyplot as plt

# 1. Corrected MuJoCo XML
# - Gravity is restored (defaults to -9.81 m/s^2)
# - Body pos is "0 0 0" so qpos perfectly matches global Z height
xml_string = """
<mujoco model="ball_and_floor">
    <worldbody>
        <geom name="floor" type="plane" size="1 1 0.1" rgba="0.8 0.9 0.8 1"/>

        <body name="ball" pos="0 0 0">
            <joint name="ball_z" type="slide" axis="0 0 1"/>
            <geom name="ball_geom" type="sphere" size="0.05" rgba="0.8 0.2 0.2 1" mass="1"/>
        </body>
    </worldbody>
</mujoco>
"""

def get_distance_force_profile(model_instance, data_instance, z_positions, ball_radius=0.05):
    distances = []
    forces = []

    for z in z_positions:
        data_instance.qpos[0] = z
        mujoco.mj_forward(model_instance, data_instance)

        dist = z - ball_radius
        distances.append(dist)

        normal_force = 0.0
        for i in range(data_instance.ncon):
            c_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model_instance, data_instance, i, c_force)
            normal_force += c_force[0]

        forces.append(normal_force)

    return distances, forces

# Initialize model
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# Sweep from +0.07m separation down to -0.01m penetration
z_vals = np.linspace(0.1, 0.04, 200)

# --- RUN 2: Contact Continuation (Smoothed Field) ---
ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")

# --- RUN 1: Default Hard Contact ---
model.geom_solimp[ball_geom_id][:] = [0.0001, 0.99, 0.001, 0.5, 8]
model.geom_solref[ball_geom_id][:]=[0.02, 1]
dist_default, force_default = get_distance_force_profile(model, data, z_vals)

# Activate detection early and shift the zero-penetration point
model.geom_margin[ball_geom_id] = 0.008
model.geom_gap[ball_geom_id] = 0.0

# Soften the contact: [dmin, dmax, width, midpoint, power]
model.geom_solimp[ball_geom_id][:] = [0.0001, 0.99, 0.02, 0.95, 8]
model.geom_solref[ball_geom_id][:] = [0.02, 2]

dist_smooth, force_smooth = get_distance_force_profile(model, data, z_vals)

# 2. Plotting
plt.figure(figsize=(10, 6))

default = plt.plot(dist_default, force_default, linewidth=5, linestyle='--',)# label="Discontinuous Contact Force")
smooth = plt.plot(dist_smooth, force_smooth, linewidth=5, linestyle='--',)# label="Continuous Contact Force")
contact = plt.axvline(0, color='black', linewidth=7.5,)# label="Contact Surface")
plt.legend([default, smooth, contact], ["Default Contact", "Continuous Contact", "Contact Surface"], fontsize=30, loc="upper right")

plt.xlabel('Signed Distance', fontsize=30)
plt.ylabel('Normal Force', fontsize=30)
plt.xticks([])
plt.yticks([])
plt.title('Continuous vs. Discontinuous Contact', fontsize=30)
plt.xlim(-0.01, 0.01)
plt.ylim(-0.5, 30.0)
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()

plt.show()
