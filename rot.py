import numpy as np

# Rotate about x (roll)
def R_x(o):
    R = np.array([
        [1, 0, 0],
        [0, np.cos(o), -np.sin(o)],
        [0, np.sin(o), np.cos(o)],
    ])
    return np.around(R, decimals=3)

# Rotate about y (pitch)
def R_y(o):
    R = np.array([
        [np.cos(o), 0, np.sin(o)],
        [0, 1, 0],
        [-np.sin(o), 0, np.cos(o)]
    ])
    return np.around(R, decimals=3)

# Rotate about z (yaw)
def R_z(o):
    R = np.array([
        [np.cos(o), -np.sin(o), 0],
        [np.sin(o), np.cos(o), 0],
        [0, 0, 1],
    ])
    return np.around(R, decimals=3)

# Create homogenous transformation matrix from SO3 and translation vector
def T(R, p):
    t = np.zeros((4,4))
    t[:3,:3] = R
    t[:3,3] = np.array(p).T
    t[3] = [0, 0, 0, 1]
    return np.around(t, decimals=3)

# Create skew symmetric (so3) matrix from 3d vector v
def so3(v):
    assert len(v) == 3
    w = np.zeros((3,3))
    w[0][1] = -v[2]
    w[0][2] = v[1]
    w[1][2] = -v[0]
    return w - w.T

def twist(q, s, theta, pitch):
    v = np.zeros(6)
    v[:3] = np.array(s) * theta
    v[3:] = np.cross(-np.array(s) * theta, q) + pitch * np.array(s) * theta

    if np.sum(np.abs(v[:3])) > 1e-6:
        v /= np.linalg.norm(v[:3])
    else:
        raise NotImplementedError(q,s,theta,pitch)
    return v

# Create SE3 from a twist and angular displacement
def SE3(twist, theta):
    w, v = twist[:3], twist[3:]

    w_skew = so3(w)

    e_twist_t = np.zeros((4,4))

    e_twist_t[:3, :3] = np.eye(3) + np.sin(theta) * w_skew + \
                        (1 - np.cos(theta)) * (w_skew @ w_skew)

    e_twist_t[:3, 3] = (np.eye(3) * theta + (1 - np.cos(theta)) * w_skew + \
                       (theta - np.sin(theta)) * (w_skew @ w_skew)) @ v
    e_twist_t[3, 3] = 1
    return np.around(e_twist_t, decimals=3)

# Problem 3.24
t = 4

q1 = np.zeros(3)
s1 = [0, 0, 1] # Rotate about z
theta1 = np.pi/4
S1 = twist(q1, s1, theta1, 2)
e_S1 = SE3(S1, t*theta1)
print(e_S1)

q2 = np.array([0, 0, 10])
s2 = [0, 1, 0]
theta2 = np.pi/8
S2 = twist(q2, s2, theta2, 0)
e_S2 = SE3(S2, t*theta2)
print(e_S2)

q3 = np.array([0, 5, 5])
s3 = [1, 0, 0]
theta3 = np.pi/4
S3 = twist(q3, s3, theta3, 0)
e_S3 = SE3(S3, t*theta3)
print(e_S3)

T_sb = T(R_z(np.pi/2), [0, 8, 5])
print(T_sb)

T_sb_prime = e_S1 @ e_S2 @ e_S3 @ T_sb
print('final:', T_sb_prime)


#Ra = R_z(np.pi/2) @ R_x(np.pi/2)
#Ra2 = R_x(np.pi/2) @ R_z(np.pi/2)
#
#Rb = R_x(-np.pi/2)
#
#Rab_dumb = R_z(-np.pi/2) @ R_x(np.pi) @ Ra
#
#Ra = Ra2
#x = [1, 0, 0]
#assert np.sum(Ra @ x - [0, 0, 1]) < 1e-4
#
#x = [0, 0, 1]
#assert np.sum(Ra @ x - [0, -1, 0]) < 1e-4
#
#x = [0, 1, 0]
#assert np.sum(Ra @ x - [-1, 0, 0]) < 1e-4
#
#x = [1, 0, 0]
#assert np.sum(Ra.T @ x - [0, -1, 0]) < 1e-4
#
#x = [0, 1, 0]
#assert np.sum(Ra.T @ x - [0, 0, -1]) < 1e-4
#
#x = [0, 0, 1]
#assert np.sum(Ra.T @ x - [1, 0, 0]) < 1e-4
#
#print(np.linalg.det(Ra), np.linalg.det(Rb))
