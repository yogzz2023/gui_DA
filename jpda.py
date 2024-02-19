import numpy as np

# Define the state transition matrix
F = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])  # Constant velocity model

# Define the measurement matrix
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Define the measurement noise covariance matrix
R = np.eye(2) * 5  # Assuming measurement noise with covariance 5

# Define initial state covariance
P = np.eye(4) * 100

# Define process noise covariance matrix
Q = np.eye(4) * 0.1  # Assuming process noise with covariance 0.1

# Measurement
m1 = np.array([12, 22])
m2 = np.array([23, 17])

# Initialize targets
target1 = np.array([10, 20, 0, 0])  # Initial state for target 1 (x, y, vx, vy)
target2 = np.array([25, 12, 0, 0])  # Initial state for target 2 (x, y, vx, vy)

# Kalman filter predict and update functions
def predict(x, P, Q, F):
    x_pred = np.dot(F, x)
    P_pred = np.dot(np.dot(F, P), F.T) + Q
    return x_pred, P_pred

def update(x_pred, P_pred, z, R, H):
    y = z - np.dot(H, x_pred)
    S = np.dot(np.dot(H, P_pred), H.T) + R
    K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(S))
    x_updated = x_pred + np.dot(K, y)
    P_updated = P_pred - np.dot(np.dot(K, H), P_pred)
    return x_updated, P_updated

# Association probabilities
Pd = 0.9  # Probability of detection
Pfa = 0.1  # Probability of false alarm

# Calculate association probabilities
P_M = np.array([[Pd * (1 - Pfa), Pd * Pfa],
                [(1 - Pd) * (1 - Pfa), (1 - Pd) * Pfa]])

# Loop through each target
for i, target in enumerate([target1, target2], start=1):
    # Predict
    target[:4], P = predict(target[:4], P, Q, F)
    
    # Update
    for j, measurement in enumerate([m1, m2], start=1):
        x_pred, P_pred = predict(target, P, Q, F)
        x_updated, P_updated = update(x_pred, P_pred, measurement, R, H)
        association_prob = P_M[i-1, j-1]
        target[:4] = target[:4] + association_prob * (x_updated[:4] - target[:4])

    print(f"Target {i} updated position:", target[:2])
