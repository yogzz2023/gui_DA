import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView

class JPDAKalmanFilter(App):
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

    def build(self):
        # Layout for input fields
        layout_input = BoxLayout(orientation='vertical')
        layout_input.add_widget(Label(text='Target 1 (initial):'))
        self.target1_x = TextInput(multiline=False)
        layout_input.add_widget(self.target1_x)
        self.target1_y = TextInput(multiline=False)
        layout_input.add_widget(self.target1_y)

        layout_input.add_widget(Label(text='Target 2 (initial):'))
        self.target2_x = TextInput(multiline=False)
        layout_input.add_widget(self.target2_x)
        self.target2_y = TextInput(multiline=False)
        layout_input.add_widget(self.target2_y)

        layout_input.add_widget(Label(text='Pd (Prob. of Detection):'))
        self.pd = TextInput(multiline=False)
        layout_input.add_widget(self.pd)

        layout_input.add_widget(Label(text='Pfa (Prob. of False Alarm):'))
        self.pfa = TextInput(multiline=False)
        layout_input.add_widget(self.pfa)

        # Button to process data
        process_button = Button(text="Process Data")
        process_button.bind(on_press=self.process_data)
        layout_input.add_widget(process_button)

        # Output text area
        self.output_text = TextInput(text='', multiline=True)
        layout_output = ScrollView()
        layout_output.add_widget(self.output_text)

        root_layout = BoxLayout(orientation='vertical')
        root_layout.add_widget(layout_input)
        root_layout.add_widget(layout_output)

        return root_layout

    def predict(self, x, P, Q, F):
        x_pred = np.dot(self.F, x)
        P_pred = np.dot(np.dot(self.F, P), self.F.T) + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z, R, H):
        y = z - np.dot(self.H, x_pred)
        S = np.dot(np.dot(self.H, P_pred), self.H.T) + self.R
        K = np.dot(np.dot(P_pred, self.H.T), np.linalg.inv(S))
        x_updated = x_pred + np.dot(K, y)
        P_updated = P_pred - np.dot(np.dot(K, self.H), P_pred)
        return x_updated, P_updated

    def process_data(self, instance):
        try:
            target1 = np.array([float(self.target1_x.text), float(self.target1_y.text), 0, 0])
            target2 = np.array([float(self.target2_x.text), float(self.target2_y.text), 0, 0])

            # Measurements
            m1 = np.array([12, 22])
            m2 = np.array([23, 17])

            # Association probabilities
            Pd = float(self.pd.text)
            Pfa = float(self.pfa.text)

            # Calculate association probabilities
            P_M = np.array([[Pd * (1 - Pfa), Pd * Pfa],
                            [(1 - Pd) * (1 - Pfa), (1 - Pd) * Pfa]])

            # Loop through each target
            output_str = ''
            for i, target in enumerate([target1, target2], start=1):
                output_str += f"\nProcessing Target {i}:\n"
                output_str += f"Initial Position: {target[:2]}\n"

                # Predict
                target[:4], self.P = self.predict(target[:4], self.P, self.Q, self.F)
                output_str += f"Predicted Position: {target[:2]}\n"

                # Update
                for j, measurement in enumerate([m1, m2], start=1):
                    x_pred, P_pred = self.predict(target, self.P, self.Q, self.F)
                    x_updated, P_updated = self.update(x_pred, P_pred, measurement, self.R, self.H)
                    association_prob = P_M[i-1, j-1]
                    target[:4] = target[:4] + association_prob * (x_updated[:4] - target[:4])
                    output_str += f"Measurement {j}: {measurement}\n"
                    output_str += f"Association Probability: {association_prob}\n"
                    output_str += f"Updated Position: {target[:2]}\n"

                output_str += f"Final Updated Position: {target[:2]}\n"

            # Display initial values of target 1, target 2, Pd, Pfa
            output_str += "\nInitial Values:\n"
            output_str += f"Target 1 (initial): x={self.target1_x.text}, y={self.target1_y.text}\n"
            output_str += f"Target 2 (initial): x={self.target2_x.text}, y={self.target2_y.text}\n"
            output_str += f"Pd (Prob. of Detection): {self.pd.text}\n"
            output_str += f"Pfa (Prob. of False Alarm): {self.pfa.text}\n"

            self.output_text.text = output_str

        except ValueError:
            self.output_text.text = "Invalid input. Please enter numeric values."

if __name__ == '__main__':
    JPDAKalmanFilter().run()
