import cv2
import time
import threading
from queue import Queue
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError("Could not open video file")
            
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Use 'mp4v' codec for MP4 output
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(
            output_path, 
            self.fourcc, 
            self.fps, 
            (self.width, self.height))  
            
        if not self.output_video.isOpened():
            raise ValueError("Could not create video writer")
            
        self.model = YOLO('yolov8s-pose.pt')  # Load YOLOv8 Pose model
        self.lock = threading.Lock()  # Lock for thread-safe video writing

    def __del__(self):
        if hasattr(self, 'output_video') and self.output_video:
            self.output_video.release()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def read_frames(self):
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def process_frame(self, frame):
        results = self.model(frame)  # Get detection results
        annotated_frame = results[0].plot()  # Draw annotations on frame
        return annotated_frame

    def process_frames_single_thread(self):
        start_time = time.time()
        frames = self.read_frames()
        for frame in frames:
            processed_frame = self.process_frame(frame)
            self.output_video.write(processed_frame)
        end_time = time.time()
        return end_time - start_time

    def process_frames_multi_thread(self, num_threads=4):
        start_time = time.time()
        frames = self.read_frames()
        frame_queue = Queue()
        result_queue = Queue()

        # Fill the queue with frame indices and frames
        for idx, frame in enumerate(frames):
            frame_queue.put((idx, frame))

        def worker():
            while not frame_queue.empty():
                try:
                    idx, frame = frame_queue.get_nowait()
                    processed_frame = self.process_frame(frame)
                    result_queue.put((idx, processed_frame))
                except:
                    break

        # Start worker threads
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results in order
        results = [None] * len(frames)
        while not result_queue.empty():
            idx, frame = result_queue.get()
            results[idx] = frame

        # Write frames to video in correct order
        for frame in results:
            if frame is not None:
                with self.lock:
                    self.output_video.write(frame)

        end_time = time.time()
        return end_time - start_time

def main():
    input_video = 'input_video.mp4'
    output_video = 'output_video.mp4'
    mode = 'single'  # 'single' or 'multi'
    
    try:
        processor = VideoProcessor(input_video, output_video)
        
        if mode == 'single':
            processing_time = processor.process_frames_single_thread()
        else:
            processing_time = processor.process_frames_multi_thread(num_threads=8)
            
        print(f'Video processing completed in {processing_time:.2f} seconds')
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    main()