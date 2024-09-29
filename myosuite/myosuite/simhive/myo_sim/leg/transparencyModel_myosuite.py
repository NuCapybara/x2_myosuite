import mujoco
import numpy as np
import os
import mediapy as media
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import csv
import math

# This file is used to control the exoskeleton in the transparency mode to follow the human knee joint angle.
# The control signal is calculated by the PD controller on the interaction torque
# between the exoskeleton and the human.
# The motion: both two legs(hip+knee) moving in sin wave motion with the same amplitude and frequency
# The graph produces a transparency mode interaction torque on both left and right hip and knee

# Path to the XML file
xml_path = "myolegs_compatible.xml"

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


# Time duration for the motion control
t_init = 2
t_end = 10
t = []

# Containers for storing the joint angles
qact0 = []
qref0 = []
qact1 = []
qref1 = []
qact2 = []
qref2 = []
qact3 = []
qref3 = []

# Previous time step value
prev_torque_exo_lhip = None
prev_torque_exo_lknee = None
prev_torque_exo_rhip = None
prev_torque_exo_rknee = None
time_prev = 0

# The desired starting angle starts from 0 for each joint(in angle [rad])

q0_init = 0  # Initial joint angle of human knee joint (left human knee)
q1_init = 0  # Initial joint angle of human hip joint (left human hip)
q2_init = 0  # Initial joint angle of human knee joint (right human knee)
q3_init = 0  # Initial joint angle of human hip joint (right human hip)

# Containers
qact_exo_lknee_inertia = []
qact_exo_lknee = []
qact_exo_lhip = []
# human_knee_torque appends the control signal calculated to the human knee joint
human_knee_torque = []
knee_joint_smooth_force = []
knee_joint_bias_force = []
torque_calculated_lknee = []
knee_passive_force = []
exo_knee_act_force = []
exo_knee_contraint_force = []
err_exo_left_knee = []
err_exo_left_hip = []
err_exo_right_knee = []
err_exo_right_hip = []
exo_left_knee_control_signal = []
qact_exo_rknee_inertia = []
qact_exo_rknee = []

grav = []
left_foot_sensor = []
right_foot_sensor = []
# contact_force = []


# controller function to control the exoskeleton to follow the human knee joint angle
def controller(model, data):
    global prev_torque_exo_lhip, prev_torque_exo_lknee, prev_torque_exo_rhip, prev_torque_exo_rknee
    global time_prev

    time = data.time
    if time < t_init:
        return
    # User Input: user can specify the amplitude and frequency of the movement
    # Offset should be defined as the knee joints cannot go negative
    # Define the amplitude and frequency of the sine wave for the motion
    amplitude = 1.37 / 2.0  # example amplitude (can be adjusted)
    frequency = 0.5  # example frequency in Hz (can be adjusted)
    offset = 1.37 / 2.0  # example offset (can be adjusted)

    if time > t_end:
        time = t_end
    if time < t_init:
        time = t_init

    q0_ref = amplitude * np.sin(2 * np.pi * frequency * time) + offset
    q0dot_ref = 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * time)

    q1_ref = amplitude * np.sin(2 * np.pi * frequency * time) + offset
    q1dot_ref = -q0dot_ref

    q2_ref = amplitude * np.sin(2 * np.pi * frequency * time) + offset
    q2dot_ref = 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * time)

    q3_ref = amplitude * np.sin(2 * np.pi * frequency * time) + offset
    q3dot_ref = 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * time)

    # Kp and Kd values for the PD position controller
    kp = 15
    kd = 0.5
    # Kp_exo_Force is used to fix the hip joint angle of the exoskeleton
    # Should only be used when hip transparency mode is off, or they will interact each other
    kp_exo_force = 0
    actid_left_knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gasmed_l")
    # Avoid the case when actid is not found
    ### Actuate the human left knee joint
    if actid_left_knee != -1:
        data.ctrl[actid_left_knee] = kp * (q0_ref - data.qpos[qpos_left_knee]) + kd * (
            q0dot_ref - data.qvel[qpos_left_knee]
        )
    # left knee (bflh_l/bfsh_l/gaslat_l/gasmed_l->gasmed_l will not relate to the hip part, good for isolation)

    ### Actuate the human left hip joint
    actid_left_hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "psoas_l")
    data.ctrl[actid_left_hip] = kp * (q1_ref - data.qpos[qpos_left_hip]) + kd * (
        q1dot_ref - data.qvel[qpos_left_hip]
    )

    ### Actuate the left hip joint of the **exoskeleton** to keep the hip joint angle as 0
    left_exo_hip_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_joint"
    )

    qpos_left_exo_hip = model.jnt_qposadr[
        left_exo_hip_joint_id
    ]  # qpos index for the exoskeleton left hip joint

    err_exo_left_hip_qpos = (
        0 - data.qpos[qpos_left_exo_hip]
    )  # error between the current exoskeleton hip joint angle and the reference angle

    actid_left_exo_hip = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "left_hip_joint_motor",  # actuator id for the exoskeleton left hip joint
    )
    data.ctrl[actid_left_exo_hip] = (
        kp_exo_force * err_exo_left_hip_qpos
    )  # PD control on the exoskeleton left hip joint
    qact_exo_lhip.append(
        data.qpos[qpos_left_exo_hip]
    )  # append the exoskeleton left hip joint angle to the list

    ### Actuate the human right hip joint
    actid_right_hip = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "psoas_r"
    )  # actuator id for the human posoas_r muscle, for right hip joint control
    data.ctrl[actid_right_hip] = kp * (q3_ref - data.qpos[qpos_right_hip]) + kd * (
        q3dot_ref - data.qvel[qpos_right_hip]
    )  # right hip(glmax1_r/glmax2_r/glmax3_r)

    ### Actuate the human right knee joint
    actid_right_knee = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gasmed_r"
    )
    data.ctrl[actid_right_knee] = kp * (q2_ref - data.qpos[qpos_right_knee]) + kd * (
        q2dot_ref - data.qvel[qpos_right_knee]
    )  # right hip(glmax1_r/glmax2_r/glmax3_r)

    # Append to lists
    t.append(data.time)

    qact0.append(data.qpos[qpos_left_knee])  # left knee joint actual positino
    qact1.append(data.qpos[qpos_left_hip])  # left hip joint actual position
    qact2.append(data.qpos[qpos_right_knee])  # right knee actual position
    qact3.append(data.qpos[qpos_right_hip])  # right hip actual position

    ### Append the foot plate sensor data, foot plates implemented as contact force sensors
    left_foot_sensor.append(data.sensordata[0])
    right_foot_sensor.append(data.sensordata[1])

    ### Finding the joint id for the exoskeleton joint
    right_exo_knee_inertia_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee_joint_drive"
    )
    right_exo_hip_inertia_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_joint_drive"
    )
    left_exo_hip_inertia_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_joint_drive"
    )
    right_exo_hip_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_hip_joint"
    )

    right_exo_knee_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee_joint"
    )
    ### Finding the actuator id for the exoskeleton joint
    actid_right_exo_hip = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_joint_motor"
    )
    actid_right_exo_knee = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee_joint_motor"
    )
    ### Finding the qpos id for the exoskeleton joint
    qpos_exo_right_knee_inertia = model.jnt_qposadr[right_exo_knee_inertia_joint_id]
    qpos_exo_right_hip_inertia = model.jnt_qposadr[right_exo_hip_inertia_joint_id]
    qpos_exo_left_hip_inertia = model.jnt_qposadr[left_exo_hip_inertia_joint_id]
    qpos_exo_right_knee = model.jnt_qposadr[right_exo_knee_joint_id]
    ### Finding the dofadr id for the exoskeleton joint [for frc_smooth and frc_constraint functions]
    dofadr_exo_right_knee = model.jnt_dofadr[right_exo_knee_joint_id]

    ### Actuate the exoskeleton- Transparency control on
    trans_kp = 1.0
    trans_kd = 0.00001
    M_thigh = 2.1844  # mass of the exoskeleton thigh, can be found in xml file
    len_thigh = 0.37  # length of the exoskeleton thigh, can be found in xml file
    alpha_left = data.qpos[
        qpos_exo_left_hip_inertia
    ]  # angle of the exoskeleton left hip joint
    theta_left = data.qpos[
        qpos_exo_left_knee_inertia
    ]  # angle of the exoskeleton left knee joint
    alpha_right = data.qpos[
        qpos_exo_right_hip_inertia
    ]  # angle of the exoskeleton right hip joint
    theta_right = data.qpos[
        qpos_exo_right_knee_inertia
    ]  # angle of the exoskeleton right knee joint
    com_shank_to_knee = 0.312012  # distance from the shank center of mass to knee, can be found in xml file
    com_thigh_to_hip = 0.254625  # distance from the thigh center of mass to hip, can be found in xml file
    M_shank = 1.6105  # mass of the exoskeleton shank, can be found in xml file
    g = 9.81  # gravity constant

    ### Transpanrency model on left hip. PD control on interaction torque.

    # calculate the gravity torque on the exoskeleton left thigh
    bias_torque_left_thigh_calculated = (
        (math.sin(alpha_left) * len_thigh + com_shank_to_knee * math.sin(theta_left))
        * M_shank
        * g
    ) + (
        math.sin(data.qpos[qpos_exo_left_hip_inertia]) * M_thigh * g * com_thigh_to_hip
    )

    dofadr_exo_left_hip = model.jnt_dofadr[
        left_exo_hip_joint_id
    ]  # dofadr index for the exoskeleton left hip joint
    err_int_t_exo_hip_left = 0 - (
        data.qfrc_smooth[dofadr_exo_left_hip]
        + data.qfrc_constraint[dofadr_exo_left_hip]
        - bias_torque_left_thigh_calculated
    )

    # calculate the kd part(time derivative of the interaction torque error)
    dt = time - time_prev
    if dt <= 1e-9:
        print("Invalid time step detected")
        return
    else:
        time_prev = time

    if prev_torque_exo_lhip is not None:
        if prev_torque_exo_lhip == 0:
            print(
                "this is the first time in loop, the previous exo left hip torque is ,",
                prev_torque_exo_lhip,
            )
        else:
            if dt > 0:
                dt_err_int_t_exo_lhip = (
                    err_int_t_exo_hip_left - prev_torque_exo_lhip
                ) / dt

                ### PD control. kd part is calculated by （err_now - err_prev）/ dt
                # Goal:minimize torque_interacion:
                # T_int = T_exo_joint - bias_torque_calculated
                #      = qfrc_smooth + qfrc_constraint - bias_torque_calculated
                data.ctrl[actid_left_exo_hip] = (
                    trans_kp * err_int_t_exo_hip_left + trans_kd * dt_err_int_t_exo_lhip
                ) + bias_torque_left_thigh_calculated
            else:
                return  # time step is invalid
    ### record the prev torque for the next time step use
    prev_torque_exo_lhip = err_int_t_exo_hip_left
    ### record the last time to get the time difference
    time_prev = time
    ### Append the left hip interaction torque
    err_exo_left_hip.append(err_int_t_exo_hip_left)

    qact_exo_lknee_inertia.append(data.qpos[qpos_exo_left_knee_inertia])

    ### Transparency model on the left knee joint. PD control on interaction torque.
    bias_torque_calculated_left_knee = (
        math.sin(math.pi - data.qpos[qpos_exo_left_knee_inertia])
        * (M_shank)
        * g
        * com_shank_to_knee
    )
    torque_calculated_lknee.append(bias_torque_calculated_left_knee)
    qact_exo_lknee.append(data.qpos[qpos_exo_left_knee])

    actid_left_exo_knee = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_joint_motor"
    )
    # int torque reference = 0, so ref-action = 0 - (qfrc_smooth + qfrc_constraint - bias_torque_calculated)
    err_int_t_exo_lknee_ = 0 - (
        data.qfrc_smooth[dofadr_exo_left_knee]
        + data.qfrc_constraint[dofadr_exo_left_knee]
        - bias_torque_calculated_left_knee
    )

    if prev_torque_exo_lknee is not None:

        if dt > 0:
            dt_err_int_t_exo_lknee = (err_int_t_exo_lknee_ - prev_torque_exo_lknee) / dt
            data.ctrl[actid_left_exo_knee] = (
                trans_kp * err_int_t_exo_lknee_ + trans_kd * dt_err_int_t_exo_lknee
            ) + bias_torque_calculated_left_knee
        else:
            print("Ooops the time step is < 0, the dt is", dt)
            return

    prev_torque_exo_lknee = err_int_t_exo_lknee_

    # when actuating the transparency mode, we plus the bias torque calculated for the feed forward control.
    # bc when the int torque is 0, at the time, the control signal is 0 by just kp*err_int_torque, we need to compensate the bias torque,
    # so we add the bias torque calculated
    # data.ctrl[actid_left_exo_knee] = (
    #     trans_kp * err_int_t_exo_knee_ + kd * err_dot_int_t_exo_knee + bias_torque_calculated
    # )

    err_exo_left_knee.append(err_int_t_exo_lknee_)
    exo_left_knee_control_signal.append(data.ctrl[actid_left_exo_knee])

    # Transparency model on the right hip joint. PD control on interaction torque.
    bias_torque_right_thigh_calculated = (
        (math.sin(alpha_right) * len_thigh + com_shank_to_knee * math.sin(theta_right))
        * M_shank
        * g
    ) + (
        math.sin(data.qpos[qpos_exo_right_hip_inertia]) * M_thigh * g * com_thigh_to_hip
    )
    dofadr_exo_right_hip = model.jnt_dofadr[right_exo_hip_joint_id]
    err_int_t_exo_hip_right = 0 - (
        data.qfrc_smooth[dofadr_exo_right_hip]
        + data.qfrc_constraint[dofadr_exo_right_hip]
        - bias_torque_right_thigh_calculated
    )
    if prev_torque_exo_rhip is not None:
        if prev_torque_exo_rhip == 0:
            print(
                "this is the first time in loop, the previous exo right hip torque is ,",
                prev_torque_exo_rhip,
            )
        else:
            if dt > 0:
                dt_err_int_t_exo_rhip = (
                    err_int_t_exo_hip_right - prev_torque_exo_rhip
                ) / dt

                # PD control. kd part is calculated by （err_now - err_prev）/ dt
                data.ctrl[actid_right_exo_hip] = (
                    trans_kp * err_int_t_exo_hip_right
                    + trans_kd * dt_err_int_t_exo_rhip
                ) + bias_torque_right_thigh_calculated
            else:
                return  # invalid time step

    # Transparency model on the right knee joint. PD control on interaction torque.
    qact_exo_rknee_inertia.append(data.qpos[qpos_exo_right_knee_inertia])
    bias_torque_calculated_rknee = (
        math.sin(math.pi - data.qpos[qpos_exo_right_knee_inertia])
        * (M_shank)
        * g
        * com_shank_to_knee
    )
    qact_exo_rknee.append(data.qpos[qpos_exo_right_knee])

    # int torque reference = 0, so ref-action = 0 - (qfrc_smooth + qfrc_constraint - bias_torque_calculated)
    err_int_t_exo_rknee_ = 0 - (
        data.qfrc_smooth[dofadr_exo_right_knee]
        + data.qfrc_constraint[dofadr_exo_right_knee]
        - bias_torque_calculated_rknee
    )

    if prev_torque_exo_rknee is not None:

        if dt > 0:
            dt_err_int_t_exo_rknee = (err_int_t_exo_rknee_ - prev_torque_exo_rknee) / dt
            data.ctrl[actid_right_exo_knee] = (
                trans_kp * err_int_t_exo_rknee_ + trans_kd * dt_err_int_t_exo_rknee
            ) + bias_torque_calculated_rknee
        else:
            return  # invalid time step

    ### record the prev torque for the next time step using
    prev_torque_exo_rknee = err_int_t_exo_rknee_
    prev_torque_exo_rhip = err_int_t_exo_hip_right

    err_exo_right_knee.append(err_int_t_exo_rknee_)
    err_exo_right_hip.append(err_int_t_exo_hip_right)

    # Append to reference angle lists
    qref0.append(q0_ref)
    qref1.append(q1_ref)
    qref2.append(q2_ref)
    qref3.append(q3_ref)

    human_knee_torque.append(data.ctrl[actid_left_knee])

    # qfrc_smooth index depends on nv, degree of freedom
    ### Append the force data into the lists
    knee_joint_smooth_force.append(
        data.qfrc_smooth[dofadr_exo_left_knee]
    )  # the unconstraint force on the exo left knee joint
    knee_joint_bias_force.append(
        data.qfrc_bias[dofadr_exo_left_knee]
    )  # the bias force on the exo left knee joint
    knee_passive_force.append(
        data.qfrc_passive[dofadr_exo_left_knee]
    )  # the passive force on the exo left knee joint
    exo_knee_act_force.append(
        data.qfrc_actuator[dofadr_exo_left_knee]
    )  # the actuator force on the exo left knee joint
    exo_knee_contraint_force.append(
        data.qfrc_constraint[dofadr_exo_left_knee]
    )  # the constraint force on the exo left knee joint


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)


def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    button_middle = (
        glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    )
    button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if not button_left and not button_middle and not button_right:
        return

    width, height = glfw.get_window_size(window)
    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT

    if button_right:
        action = (
            mujoco.mjtMouse.mjMOUSE_MOVE_H
            if mod_shift
            else mujoco.mjtMouse.mjMOUSE_MOVE_V
        )
    elif button_left:
        action = (
            mujoco.mjtMouse.mjMOUSE_ROTATE_H
            if mod_shift
            else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        )
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
    mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll(window, xoffset, yoffset):
    action = mujoco.mjtMouse.mjMOUSE_ZOOM
    mujoco.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)


def get_sensor_sensordata():
    return mujoco.MjData(model).sensordata


# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mujoco.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mujoco.MjData(model)  # MuJoCo data
cam = mujoco.MjvCamera()  # Abstract camera
opt = mujoco.MjvOption()

for i in range(model.nu):  # model.nu is the number of actuators
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"Index {i} corresponds to actuator '{actuator_name}'")

for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"Index {i} corresponds to joint '{joint_name}'")

# human left knee joint id and qpos id
left_knee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_l")
qpos_left_knee = model.jnt_qposadr[left_knee_joint_id]
# human left hip joint id and qpos id
left_hip_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_l")
qpos_left_hip = model.jnt_qposadr[left_hip_joint_id]

# human right knee joint id and qpos id
right_knee_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_r"
)
qpos_right_knee = model.jnt_qposadr[right_knee_joint_id]

# human right hip joint id and qpos id
right_hip_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "hip_flexion_r"
)
qpos_right_hip = model.jnt_qposadr[right_hip_joint_id]

# exo left knee joint inertia id and qpos id
exo_left_knee_joint_inertia_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint_drive"
)
qpos_exo_left_knee_inertia = model.jnt_qposadr[exo_left_knee_joint_inertia_id]

# exo left knee joint id and dofadr id for frc_smooth[index]
exo_left_knee_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint"
)
dofadr_exo_left_knee = model.jnt_dofadr[exo_left_knee_joint_id]
qpos_exo_left_knee = model.jnt_qposadr[exo_left_knee_joint_id]



print("---------------------------------------------------")


print(qpos_left_knee, "the human knee left joint qposadr!")
print(qpos_left_hip, "the human hip left joint qposadr!")
print(qpos_right_knee, "the human knee right joint qposadr!")
print(qpos_right_hip, "the human hip right joint qposadr!")
print(
    dofadr_exo_left_knee,
    "the EXO knee left joint dofadr! For the frc_smooth[] using! ",
    exo_left_knee_joint_id,
    "the EXO knee left joint id! ",
)
print(
    qpos_exo_left_knee_inertia,
    "the EXO knee left joint inertia qposadr!",
)
print(qpos_exo_left_knee, "the EXO knee left joint qposadr! ")
# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)
scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Set camera configuration
cam.azimuth = 90
cam.elevation = 5
cam.distance = 6
cam.lookat = np.array([0.0, 0.0, 0.0])

simend = t_end  # simulation time

# Initialize the controller
data.qpos[qpos_left_knee] = q0_init
data.qpos[qpos_left_hip] = q1_init
data.qpos[qpos_right_knee] = q2_init
data.qpos[qpos_right_hip] = q3_init

mujoco.set_mjcb_control(controller)


# Simulation parameters

with open("sensor_data.csv", mode="a") as file:
    writer = csv.writer(file)
    while not glfw.window_should_close(window):
        time_prev_sim = data.time

        while data.time - time_prev_sim < 1.0 / 60.0:
            mujoco.mj_step(model, data)
            writer.writerow(data.sensordata)

        # print(knee_joint_passive_force)
        if data.time >= simend:
            min_length = min_length = min(len(t), len(qref0), len(qact0))
            # Ensure all lists are truncated to min_length

            t = t[:min_length]
            knee_joint_smooth_force = knee_joint_smooth_force[:min_length]
            knee_joint_bias_force = knee_joint_bias_force[:min_length]
            human_knee_torque = human_knee_torque[:min_length]

            plt.figure(1)
            plt.subplot(2, 1, 1)
            # plt.plot(t, np.subtract(qref0[:min_length], qact0[:min_length]), "k")
            # plt.plot(t, qref0[:min_length], "r")
            # plt.plot(t, qact0[:min_length], "b")
            # plt.legend(["error", "reference_left_knee_position", "actual_left_knee_position"])
            # plt.ylabel("position/angle (rad)")

            # plt.subplot(6, 1, 2)
            # plt.plot(t, np.subtract(qref1[:min_length], qact1[:min_length]), "k")
            # plt.plot(t, qref1[:min_length], "r")
            # plt.plot(t, qact1[:min_length], "b")

            # plt.legend(["error", "reference_left_hip_position", "actual_left_hip_position"])
            # plt.ylabel("position/angle (rad)")

            # plt.subplot(6, 1, 3)
            # # plt.plot(t, qact_exo_lknee, "g-")
            # # plt.plot(t, qact_exo_lknee_inertia, "b-")
            # plt.plot(t, np.subtract(qref2[:min_length], qact2[:min_length]), "k")
            # plt.plot(t, qref2[:min_length], "r")
            # plt.plot(t, qact2[:min_length], "b")

            # plt.legend(
            #     [
            #         "error",
            #         "reference_right_knee_position",
            #         "actual_right_knee_position",
            #     ]
            # )
            # plt.ylabel("position/angle (rad)")
            # # plt.plot(t, mujoco.mju_sub(knee_joint_smooth_force, knee_joint_bias_force, model.nv), "y")

            # plt.subplot(6, 1, 4)
            # plt.plot(t, np.subtract(qref3[:min_length], qact3[:min_length]), "k")
            # plt.plot(t, qref3[:min_length], "r")
            # plt.plot(t, qact3[:min_length], "b")

            # plt.legend(
            #     [
            #         "error",
            #         "reference_right_hip_position",
            #         "actual_right_hip_position",
            #     ]
            # )
            # plt.ylabel("position/angle (rad)")

            plt.subplot(2, 1, 1)
            plt.plot(t, err_exo_left_knee[:min_length], "r")
            plt.plot(t, err_exo_left_hip[:min_length], "b")
            plt.legend(
                [
                    "interaction_torque_left_knee",
                    "interaction_torque_left_hip",
                    # "exo_left_knee_control_signal"
                ]
            )
            plt.ylabel("Torque(Nm)")

            plt.subplot(2, 1, 2)
            plt.plot(t, err_exo_right_knee[:min_length], "r")
            plt.plot(t, err_exo_right_hip[:min_length], "b")
            plt.legend(
                [
                    "interaction_torque_right_knee",
                    "interaction_torque_right_hip",
                ]
            )
            plt.ylabel("Torque(Nm)")
            mean_rknee_int_torque = np.mean(err_exo_right_knee)
            mean_lknee_int_torque = np.mean(err_exo_left_knee)
            std_rknee_int_torque = np.std(err_exo_right_knee)
            std_lknee_int_torque = np.std(err_exo_left_knee)

            mean_rhip_int_torque = np.mean(err_exo_right_hip)
            mean_lhip_int_torque = np.mean(err_exo_left_hip)
            std_rhip_int_torque = np.std(err_exo_right_hip)
            std_lhip_int_torque = np.std(err_exo_left_hip)
            print("---------------------------------------------------")
            print("the mean right knee interaction torque and std", mean_rknee_int_torque,  ", " ,std_rknee_int_torque)
            print("the mean left knee interaction torque and std", mean_lknee_int_torque, ", " ,std_lknee_int_torque)
            print("the mean right hip interaction torque and std", mean_rhip_int_torque, ", " ,std_rhip_int_torque)
            print("the mean left hip interaction torque and std", mean_lhip_int_torque, ", " ,std_lhip_int_torque)

            plt.show(block=True)
            
            break
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        mujoco.mjv_updateScene(
            model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene
        )
        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

glfw.terminate()
