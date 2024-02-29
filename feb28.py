import numpy as np

class Measurement:
    def __init__(self, range_, azimuth, elevation, time):
        self.range = range_
        self.azimuth = azimuth
        self.elevation = elevation
        self.time = time

class JPDAFilter:
    def __init__(self, measurement):
        self.measurement = measurement
        self.initialize_filter()

    def initialize_filter(self):
        # Initial state estimate
        self.x = np.array([[self.measurement.range], [self.measurement.azimuth], [self.measurement.elevation], [self.measurement.time]])

        # Initial state covariance
        self.P = np.diag([1, 1, 1, 1])  # Diagonal covariance matrix

        # Process noise covariance (Q)
        self.Q = np.diag([0.1, 0.1, 0.1, 0.1])  # Diagonal covariance matrix

        # Measurement noise covariance (R)
        self.R = np.diag([1, 1, 1, 1])  # Diagonal covariance matrix

        # State transition matrix (A)
        self.A = np.eye(4)  # Identity matrix

        # Measurement matrix (H)
        self.H = np.eye(4)  # Identity matrix

    def associate(self):
        # For simplicity, let's assume a single track for each measurement
        self.associated_track = self.measurement

    def update_state(self):
        # Kalman gain calculation
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

        # Measurement update
        z = np.array([[self.measurement.range], [self.measurement.azimuth], [self.measurement.elevation], [self.measurement.time]])
        y = z - self.H.dot(self.x)
        self.x = self.x + K.dot(y)

        # Covariance update
        self.P = (np.eye(4) - K.dot(self.H)).dot(self.P)

    def predict_next_state(self):
        # State prediction
        self.x = self.A.dot(self.x)
        
        # Covariance prediction
        self.P = self.A.dot(self.P).dot(self.A.T) + self.Q

def read_config_file(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split(',')
            range_ = float(data[0])
            azimuth = float(data[1])
            elevation = float(data[2])
            time = float(data[3])
            measurements.append(Measurement(range_, azimuth, elevation, time))
    return measurements

def main():
    config_file_path = "config.txt"  # Adjust this path accordingly
    measurements = read_config_file(config_file_path)

    jpda_filters = []
    for measurement in measurements:
        jpda = JPDAFilter(measurement)
        jpda.associate()
        print("Associated Track State:", jpda.x)
        
        jpda.update_state()
        print("Updated Track State:", jpda.x)
        
        jpda.predict_next_state()
        print("Predicted Track State:", jpda.x)
        
        jpda_filters.append(jpda)

if __name__ == "__main__":
    main()
