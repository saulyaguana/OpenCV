import cv2
import numpy as np


class ExceptError(Exception):
    pass

class MotionVideo:
    def __init__(self, path=0, kernel_size=(3, 3)):
        self.path = path
        self.kernel_size = kernel_size
        
    def validate_path(self):
        video = cv2.VideoCapture(self.path)
        if not video.isOpened():
            raise ExceptError("The path of your video could not be found, plese check again.")
        return video
    
    def video_properties(self):
        video = self.validate_path()
        frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        
        size = (frame_w, frame_h)
        video_out = cv2.VideoWriter("Color_output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        
        return video_out
    
    def process_video(self, color_rectangle=(0, 0, 255)):
        knn = cv2.createBackgroundSubtractorKNN(history=400)
        win_name = "Color Frames"
        win_binary = "Binary Frames"
        cv2.namedWindow(win_name)
        cv2.namedWindow(win_binary)
        video = self.validate_path()
        while True:
            has_frame, frame = video.read()
            
            if not has_frame:
                break
            
            # Stage 1: Motion based on foreground mask.
            fg_mask = knn.apply(frame)
            fg_mask_eroded = cv2.erode(fg_mask, np.ones(self.kernel_size, np.uint8))
            motion_area_eroded = cv2.findNonZero(fg_mask_eroded)
            xe, ye, we, he = cv2.boundingRect(motion_area_eroded)
            
            if motion_area_eroded is not None:
                cv2.rectangle(frame, (xe, ye), (xe + we, ye + he), color_rectangle, thickness=6)
                
            color_frame = cv2.cvtColor(fg_mask_eroded, cv2.COLOR_GRAY2BGR)
            
            cv2.imshow(win_name, frame)
            cv2.imshow(win_binary, color_frame)
            key = cv2.waitKey(1)
            
            if key == ord("Q") or key == ord("q") or key == 27:
                break
        
        video.release()
        cv2.destroyAllWindows(win_name)
        