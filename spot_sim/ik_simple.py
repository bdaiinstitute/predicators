import random
import numpy as np
from transformations import translation_matrix, rotation_matrix, inverse_matrix, concatenate_matrices

twoPi = 2*np.pi

# shoulder to base (0, 0, 0) - that is height above floor
SHOULDER_OFFSET = inverse_matrix(translation_matrix([0.292,  0.      ,  0.873]))

# IK_FRAME = 'arm_link_wr1'
#     <origin rpy="-1.5707 0 1.5707" xyz="0.25 0.0 0.0" />
TRANSFORM_TO_IK_FRAME = \
        inverse_matrix(
            concatenate_matrices(
                # Displacement to finger end (approx)
                translation_matrix((0.25, 0., 0.0)),
                # Flip coordinate frame
                rotation_matrix(np.pi/2, (0.,0.,1.)),
                rotation_matrix(-np.pi/2, (1.,0.,0.)),
                       ))

# This is a 3R elbow manipulator with a wrist with 3 intersecting axes
# That is, the simplest IK in the universe, solution below modeled on Hauser's boodk.

def IK2R(L1, L2, x, y):
    xy_sq = x**2 + y**2
    c2 = (xy_sq - L1**2 - L2**2)/(2*L1*L2)
    if abs(c2) > 1:
        return None
    elif c2 == 1.0:
        return [(np.arctan2(y,x), 0)]
    elif c2 == -1.0 and xy_sq != 0.:
        return [(np.arctan2(y,x), np.pi)]
    elif c2 == -1.0 and xy_sq == 0.:
        return [(q1, np.pi) for q1 in [0, 2*np.pi]]
    else:
        q2_1 = np.arccos(c2)
        q2_2 = -q2_1
        theta = np.arctan2(y, x)
        q1q2 = [(theta - np.arctan2(L2*np.sin(q2_i), L1+L2*np.cos(q2_i)), q2_i) \
                for q2_i in (q2_1, q2_2)]
        for q1,q2 in q1q2:
            xk = (L1*np.cos(q1) + L2*np.cos(q1 + q2))
            yk = (L1*np.sin(q1) + L2*np.sin(q1 + q2))
            # print(f'xk={xk}, yk={yk}')
            assert abs(x - xk) < 0.0001
            assert abs(y - yk) < 0.0001
        return q1q2
        
def analytic_spot_ik_6(wrist_pose_rel_shoulder, min_limits, max_limits):
    px, py, pz = wrist_pose_rel_shoulder[:3,3]   # wrist position
    l2 = 0.3385
    l3 = np.sqrt(0.40330**2 + 0.0750**2)
    q3_off = np.arctan2(0.0750, 0.40330)
    # Solve the first three joints based on position
    q123_sols = []
    xl = np.sqrt(px**2 + py**2)
    q23 = IK2R(l2, l3, xl, -pz)
    if q23 is None:
        return []
    q1 = np.arctan2(py,px)
    for (q2, q3) in q23:
        q123_sols.append((q1, q2, q3 + q3_off))
    q23 = IK2R(l2, l3, -xl, -pz)
    assert q23 is not None
    q1 = q1 + np.pi
    for (q2, q3) in q23:
        q123_sols.append((q1, q2, q3 + q3_off))
    # Solve for the wrist angles
    qfull_sols = []
    for (q1, q2, q3) in q123_sols:
        r3_inv = inverse_matrix(
                  concatenate_matrices(rotation_matrix(q1, (0,0,1)),
                                       rotation_matrix(q2+q3, (0,1,0))))
        W = concatenate_matrices(r3_inv, wrist_pose_rel_shoulder)
        q5 = np.arccos(W[0,0])
        for q in (q5, -q5):
            s5 = np.sin(q)
            q4 = np.arctan2(W[1,0]/s5, -W[2,0]/s5)
            q6 = np.arctan2(W[0,1]/s5, W[0,2]/s5)            
            angles = (q1, q2, q3, q4, q, q6)
            if all_between(min_limits, angles, max_limits):
                qfull_sols.append(angles)
    return qfull_sols

def analytic_ik(tool_pose_rel_base, min_limits, max_limits,
                nearby_conf_angles=None, max_attempts=10, **kwargs):
    """Calls analytic_spot_ik to find a list of solutions.  Then pick
    solution closest to nearby_conf_angles. 
    """
    wrist_pose_rel_base = concatenate_matrices(tool_pose_rel_base, TRANSFORM_TO_IK_FRAME)
    wrist_pose_rel_shoulder = concatenate_matrices(SHOULDER_OFFSET, wrist_pose_rel_base)

    solutions = analytic_spot_ik_6(wrist_pose_rel_shoulder, min_limits, max_limits)

    return select_solution(solutions, get_l1_distance,
                           nearby_conf=nearby_conf_angles)

#####################################
# Basic utilies
#####################################

def angle_diff(x, y):
    z = (x - y)%twoPi
    return z - twoPi if z > np.pi else z

def get_l1_distance(sol1, sol2):   # max angle difference
    maxDiff = 0.0
    diff = 0.0
    for i in range(len(sol1)):
        diff = abs(angle_diff(sol1[i], sol2[i]))
        if diff > maxDiff:
            maxDiff = diff
    return maxDiff

def all_between(lower_limits, values, upper_limits):
    assert len(lower_limits) == len(values)
    assert len(values) == len(upper_limits)
    return np.less_equal(lower_limits, values).all() and \
           np.less_equal(values, upper_limits).all()

def select_solution(solutions, get_distance, nearby_conf_angles=None, **kwargs):
    if not solutions:
        return None
    if nearby_conf_angles is None:
        return random.choice(solutions)
    return min(solutions, key=lambda conf: get_distance(nearby_conf_angles, conf, **kwargs))
