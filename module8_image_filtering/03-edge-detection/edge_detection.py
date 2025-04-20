import cv2
s = 0  # Specify 0 for accessing the web camera.
source = cv2.VideoCapture(s)

# Create a window to display the video stream.
win_name = "Filter Demo"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

PREVIEW = 0
CANNY = 1
image_filter = PREVIEW
result = None

while True:
    has_frame, frame = cv2.read(source)
    if not has_frame:
        break
    # Flip video frame for convenience.
    frame = cv2.flip(frame, 1)
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 145, 150)
    cv2.imshow(win_name, result)
    
    key = cv2.waitKey(1)
    if key == ord("Q") or key == ord("q") or key == 27:
        break
    elif key == ord("C") or key == ord("c"):
        image_filter = CANNY
    elif key == ord("P") or key == ord("p"):
        img_filter = PREVIEW 