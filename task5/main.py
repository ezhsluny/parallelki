import argparse
import time
import threading
import queue
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

class VideoProcessor:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Failed to open video file")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )

        if not self.writer.isOpened():
            raise ValueError("Failed to create video writer")
        
        self.model = YOLO('yolov8s-pose.pt')
        self.lock = threading.Lock()
        
    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'writer'):
            self.writer.release()
            
    def process_frame(self, frame):
        results = self.model(frame)
        annotated_frame = results[0].plot()

        return annotated_frame
        
    def process_single_thread(self):
        start_time = time.time()
        
        for _ in range(self.frame_count):
            ret, frame = self.cap.read()
            if not ret:
                break
                
            processed_frame = self.process_frame(frame)
            self.writer.write(processed_frame)
            
        elapsed_time = time.time() - start_time

        return elapsed_time
        
    def process_multi_thread(self, num_threads):
        if num_threads <= 0:
            raise ValueError("Number of threads must be positive")

        start_time = time.time()
        input_queue = queue.Queue(maxsize=num_threads * 2)
        output_queue = queue.Queue()
        stop_flag = threading.Event()

        # Pre-allocate frame buffer
        frame_buffer = [None] * self.frame_count
        next_frame_to_write = 0

        def worker():
            while not stop_flag.is_set():
                try:
                    frame_idx, frame = input_queue.get(timeout=1)
                    processed_frame = self.process_frame(frame)
                    output_queue.put((frame_idx, processed_frame))
                    input_queue.task_done()
                except queue.Empty:
                    if stop_flag.is_set():
                        break
                except Exception as e:
                    print(f"Error in worker thread: {e}")
                    break

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        try:
            frame_idx = 0
            while frame_idx < self.frame_count or next_frame_to_write < self.frame_count:
                # Read frames
                while frame_idx < self.frame_count and input_queue.qsize() < num_threads * 2:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    input_queue.put((frame_idx, frame))
                    frame_idx += 1

                # Process output frames
                while not output_queue.empty():
                    idx, processed_frame = output_queue.get()
                    frame_buffer[idx] = processed_frame

                    # Write frames in order
                    while next_frame_to_write < self.frame_count and frame_buffer[next_frame_to_write] is not None:
                        self.writer.write(frame_buffer[next_frame_to_write])
                        frame_buffer[next_frame_to_write] = None
                        next_frame_to_write += 1

                time.sleep(0.01)
        finally:
            stop_flag.set()
            for t in threads:
                t.join()

        elapsed_time = time.time() - start_time
        return elapsed_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str)
    parser.add_argument('--mode', type=str, choices=['single', 'multi'])
    parser.add_argument('output_path', type=str)
    parser.add_argument('--threads', type=int)
    
    args = parser.parse_args()
    
    try:
        processor = VideoProcessor(args.video_path, args.output_path)
        
        if args.mode == 'single':
            print("Single processing...")
            elapsed_time = processor.process_single_thread()
        else:
            print(f"{args.threads} threads processing...")
            elapsed_time = processor.process_multi_thread(args.threads)
            
        print(f"Processing time: {elapsed_time:.2f} s")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()