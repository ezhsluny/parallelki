import cv2
import threading
import time
import logging
import argparse
from queue import Queue


logging.basicConfig(
    filename='errors.log',
    level=logging.ERROR,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")

class SensorCam(Sensor):
    def __init__(self, camera_name: str, resolution: tuple):
        self.camera_name = camera_name
        self.resolution = resolution
        self.cap = None

        if isinstance(camera_name, int) or camera_name.isdigit():
            self.cap = cv2.VideoCapture(int(camera_name))
        else:
            self.cap = cv2.VideoCapture(camera_name)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logging.error(f"All camera opening attempts failed. Tried: {camera_name} and 0")
                raise ValueError("Cannot open any camera device.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    def get(self):
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Failed to read from camera.")
            return None
        return frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

class SensorX(Sensor):
    def __init__(self, delay: float):
        self.delay = delay 
        self._data = 0

    def get(self):
        time.sleep(self.delay)
        self._data += 1
        return self._data

class WindowImage:
    def __init__(self, window_name: str = "Sensor Data"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def show(self, img):
        if img is not None:
            cv2.imshow(self.window_name, img)

    def destroy(self):
        cv2.destroyAllWindows()

class DataCollector:
    def __init__(self, camera_name: str, resolution: tuple):
        try:
            self.camera = SensorCam(camera_name, resolution)
            self.sensors = [
                SensorX(0.01),  # 100Hz sensor
                SensorX(0.1),   # 10Hz sensor
                SensorX(1)      # 1Hz sensor
            ]
            self.image_window = WindowImage()

            self.frame_queue = Queue(maxsize=1)  # Holds latest camera frame
            self.data_queues = [Queue() for _ in self.sensors]
            self.latest_data = [0] * len(self.sensors)

            self.running = True
            self.last_update_time = time.time()
            
            self.threads = []
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            raise

    # Start all worker threads
    def start(self):
        self.threads = [
            threading.Thread(target=self.collect_camera_data, daemon=True),
            *[threading.Thread(target=self.collect_sensor_data, args=(sensor, i), daemon=True) 
              for i, sensor in enumerate(self.sensors)]
        ]
        
        for thread in self.threads:
            thread.start()

    def collect_camera_data(self):
        while self.running:
            try:
                frame = self.camera.get()
                if frame is not None:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
            except Exception as e:
                logging.error(f"Camera capture error: {str(e)}")
                time.sleep(0.1)

    def collect_sensor_data(self, sensor: SensorX, sensor_idx: int):
        while self.running:
            try:
                data = sensor.get() 
                self.data_queues[sensor_idx].put(data)
                self.latest_data[sensor_idx] = data
            except Exception as e:
                logging.error(f"Sensor {sensor_idx} error: {str(e)}")
                time.sleep(0.1)

    def update_display(self):
        current_time = time.time()
        if current_time - self.last_update_time < 1:  # Limit to 1 FPS
            return False
            
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()

                for i in range(len(self.sensors)):
                    while not self.data_queues[i].empty():
                        self.latest_data[i] = self.data_queues[i].get()

                    cv2.putText(frame, f'Sensor {i}: {self.latest_data[i]}', (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                self.image_window.show(frame)
                self.last_update_time = current_time
                return True
            return False
        except Exception as e:
            logging.error(f"Display error: {str(e)}")
            return False

    def stop(self):
        self.running = False
        self.camera.release()
        self.image_window.destroy()

def list_available_cameras(max_to_test=5):
    available = []
    for i in range(max_to_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Collector Application')
    parser.add_argument('--camera', type=str, default='0', help='Camera name or index')
    parser.add_argument('--width', type=int, default=640, help='Width of the camera resolution')
    parser.add_argument('--height', type=int, default=480, help='Height of the camera resolution')

    args = parser.parse_args()

    cameras = list_available_cameras()
    if not cameras:
        print("No cameras found! Trying default index 0 anyway...")
        cameras = [0]

    print(f"Available cameras: {cameras}")

    CAMERA_NAME = args.camera
    RESOLUTION = (args.width, args.height)

    try:
        collector = DataCollector(CAMERA_NAME, RESOLUTION)
        print("Data collector started. Press 'q' to quit.")
        collector.start()

        while True:
            collector.update_display()
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit on 'q' key
                break
    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"Main execution error: {str(e)}")
    finally:
        if 'collector' in locals():
            collector.stop()
        print("Program terminated.")
