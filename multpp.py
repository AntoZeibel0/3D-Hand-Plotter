import cv2 
import multiprocessing
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import mediapipe as mp
from matplotlib import pyplot as plt
import numpy as np

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils

DATA_PATH = os.path.join('/HDD-1T/TEST/')

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connection

def video_capturer(rh_results_queue : multiprocessing.Queue, lh_results_queue : multiprocessing.Queue, face_results_queue : multiprocessing.Queue, end_event : multiprocessing.Event):
    capture = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while capture.isOpened():
            ret, frame = capture.read()

            image, results = mediapipe_detection(frame, holistic)

            face_results_queue.put(results.face_landmarks)
            lh_results_queue.put(results.left_hand_landmarks)
            rh_results_queue.put(results.right_hand_landmarks)    

            draw_landmarks(image, results)
            cv2.imshow('OpenCV Raw Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    
    capture.release()
    end_event.set()

# Use weak perspective/linear transformation using a scaling factor d (chosen experimentally) 
def perspective_projection(x,y,z):
    d=100
    x_perspective = x / (1 + z / d)
    y_perspective = y / (1 + z / d)
    return (x_perspective, y_perspective)

# Drop the 'z' without further computations
def ortographic_projection(x,y,z):
    return (x,y)

# Helping function for plotting the 2D coordinates
def plot_hand_landmarks_2d_aux(transformed_landmarks, left):

    # Define connections between landmarks
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                   (5, 9),(9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
                   (0, 17), (13, 17), (17, 18), (18, 19), (19, 20)]
    im_size = (480, 640, 3)
    image = np.zeros(im_size, dtype=np.uint8)

    for (x, y, label) in transformed_landmarks:
        dx, dy = int(x * im_size[1]), int(y * im_size[0])
        point_str = '({:.2f},{:.2f})'.format(x, y)
        cv2.circle(image, (dx, dy), 5, (0, 255, 0), -1)
        cv2.putText(image, str(label), (dx-5, dy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, str(point_str), (dx+5, dy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

    # Draw lines to connect the landmarks
    for connection in connections:
        start_index, end_index = connection
        if start_index < len(transformed_landmarks) and end_index < len(transformed_landmarks):
            start_point = (int(transformed_landmarks[start_index][0] * im_size[1]),
                           int(transformed_landmarks[start_index][1] * im_size[0]))
            end_point = (int(transformed_landmarks[end_index][0] * im_size[1]),
                         int(transformed_landmarks[end_index][1] * im_size[0]))
            cv2.line(image, start_point, end_point, (255, 0, 0), 2) 

    if left is True:
        cv2.imshow('Left hand plot', image)
    else:
        cv2.imshow('Right hand plot', image)
    cv2.waitKey(1)

def left_hand_plotter_2d(left_hand_queue : multiprocessing.Queue):
    while True:
        results = left_hand_queue.get()
        
        transformed_lh_landmarks = []

        if results:
            lh = [(lmk.x, lmk.y, lmk.z) for lmk in results.landmark]  # Extracting x, y, z coordinates
        else:
            lh = []
                    
        landmark_no = 0
        if lh is not None:
            if lh is not []:
                for landmark in lh:
                    x, y, z = landmark
                    x_2d, y_2d =  perspective_projection(x,y,z)
                    point_landmark = (x_2d, y_2d, landmark_no)
                    transformed_lh_landmarks.append(point_landmark)
                    landmark_no+=1
                plot_hand_landmarks_2d_aux(transformed_lh_landmarks, True)

def right_hand_plotter_2d(right_hand_queue : multiprocessing.Queue):
    while True:
        results = right_hand_queue.get()
        
        transformed_rh_landmarks = []

        if results:
            rh = [(lmk.x, lmk.y, lmk.z) for lmk in results.landmark]  # Extracting x, y, z coordinates
        else:
            rh = []
                    
        landmark_no = 0

        if rh is not None:
            if rh is not []:
                for landmark in rh:
                    x, y, z = landmark
                    x_2d, y_2d =  perspective_projection(x,y,z)
                    point_landmark = (x_2d, y_2d, landmark_no)
                    transformed_rh_landmarks.append(point_landmark)
                    landmark_no+=1
                plot_hand_landmarks_2d_aux(transformed_rh_landmarks, False)

if __name__ == '__main__':
    lh_results_queue = multiprocessing.Queue()
    rh_results_queue = multiprocessing.Queue()
    face_results_queue = multiprocessing.Queue()
    end_event = multiprocessing.Event()

    video_capturer_process = multiprocessing.Process(target=video_capturer, args=(rh_results_queue, lh_results_queue, face_results_queue, end_event))
    lh_processor_2d = multiprocessing.Process(target=left_hand_plotter_2d, args=(lh_results_queue,))
    rh_processor_2d = multiprocessing.Process(target=right_hand_plotter_2d, args=(rh_results_queue,))

    video_capturer_process.start()
    lh_processor_2d.start()
    rh_processor_2d.start()

    end_event.wait()

    video_capturer_process.kill()
    lh_processor_2d.kill()
    rh_processor_2d.kill()

    cv2.destroyAllWindows()