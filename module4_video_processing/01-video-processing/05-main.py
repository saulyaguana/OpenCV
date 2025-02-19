from motion_video import MotionVideo

path = 1
kernel_size = (5, 5)

mt_video = MotionVideo(path, kernel_size)

#mt_video.video_properties()
mt_video.process_video()