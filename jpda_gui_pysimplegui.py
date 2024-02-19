import PySimpleGUI as sg
import numpy as np

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

def main():
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

    # Layout for input fields
    layout_input = [
        [sg.Text("Target 1 (initial):")],
        [sg.Text("x:"), sg.InputText(size=(10, 1), key='-TARGET1_X-'), sg.Text("y:"), sg.InputText(size=(10, 1), key='-TARGET1_Y-')],
        [sg.Text("Target 2 (initial):")],
        [sg.Text("x:"), sg.InputText(size=(10, 1), key='-TARGET2_X-'), sg.Text("y:"), sg.InputText(size=(10, 1), key='-TARGET2_Y-')],
        [sg.Text("Pd (Prob. of Detection):"), sg.InputText(size=(10, 1), key='-PD-')],
        [sg.Text("Pfa (Prob. of False Alarm):"), sg.InputText(size=(10, 1), key='-PFA-')],
        [sg.Button("Process Data", size=(15, 1), key='-PROCESS-')]
    ]

    # Layout for output
    layout_output = [
        [sg.Output(size=(80, 20), background_color='black', text_color='white', font='Courier')]
    ]

    # Create the main window
    layout = [
        [sg.Column(layout_input, element_justification='center')],
        [sg.Column(layout_output, element_justification='center')]
    ]
    window = sg.Window('JPDA Kalman Filter', layout, background_color='lightgray')

    # Event loop
    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        if event == '-PROCESS-':
            output_str = ''
            try:
                target1 = np.array([float(values['-TARGET1_X-']), float(values['-TARGET1_Y-']), 0, 0])
                target2 = np.array([float(values['-TARGET2_X-']), float(values['-TARGET2_Y-']), 0, 0])

                # Measurements
                m1 = np.array([12, 22])
                m2 = np.array([23, 17])

                # Association probabilities
                Pd = float(values['-PD-'])
                Pfa = float(values['-PFA-'])

                # Calculate association probabilities
                P_M = np.array([[Pd * (1 - Pfa), Pd * Pfa],
                                [(1 - Pd) * (1 - Pfa), (1 - Pd) * Pfa]])

                # Loop through each target
                for i, target in enumerate([target1, target2], start=1):
                    output_str += f"\nProcessing Target {i}:\n"
                    output_str += f"Initial Position: {target[:2]}\n"

                    # Predict
                    target[:4], P = predict(target[:4], P, Q, F)
                    output_str += f"Predicted Position: {target[:2]}\n"

                    # Update
                    for j, measurement in enumerate([m1, m2], start=1):
                        x_pred, P_pred = predict(target, P, Q, F)
                        x_updated, P_updated = update(x_pred, P_pred, measurement, R, H)
                        association_prob = P_M[i-1, j-1]
                        target[:4] = target[:4] + association_prob * (x_updated[:4] - target[:4])
                        output_str += f"Measurement {j}: {measurement}\n"
                        output_str += f"Association Probability: {association_prob}\n"
                        output_str += f"Updated Position: {target[:2]}\n"

                    output_str += f"Final Updated Position: {target[:2]}\n"

                # Display initial values of target 1, target 2, Pd, Pfa
                output_str += "\nInitial Values:\n"
                output_str += f"Target 1 (initial): x={values['-TARGET1_X-']}, y={values['-TARGET1_Y-']}\n"
                output_str += f"Target 2 (initial): x={values['-TARGET2_X-']}, y={values['-TARGET2_Y-']}\n"
                output_str += f"Pd (Prob. of Detection): {values['-PD-']}\n"
                output_str += f"Pfa (Prob. of False Alarm): {values['-PFA-']}\n"

            except ValueError:
                output_str += "Invalid input. Please enter numeric values."
            print(output_str)

    window.close()

if __name__ == '__main__':
    main()
