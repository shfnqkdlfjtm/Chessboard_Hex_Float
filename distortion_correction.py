import cv2 as cv
import numpy as np
from distortion_correction import calibrate_camera, undistort_image

def draw_ar(video_file, K, dist_coeff, board_pattern, board_cellsize):
    # Open a video
    video = cv.VideoCapture(r"C:\Users\samsung\Downloads\20240401_184354_854x480.mp4")

    # Prepare a 3D box for simple AR
    box_lower = board_cellsize * np.array([[4, 2, 0], [5, 2, 0], [5, 4, 0], [4, 4, 0]])
    box_upper = board_cellsize * np.array([[4, 2, -1], [5, 2, -1], [5, 4, -1], [4, 4, -1]])

    # Prepare 3D points on a chessboard
    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

    # Run pose estimation
    while True:
        # Read an image from the video
        valid, img = video.read()
        if not valid:
            break

        # Undistort the image
        undistorted_img = undistort_image(img, K, dist_coeff)

        # Estimate the camera pose
        complete, img_points = cv.findChessboardCorners(undistorted_img, board_pattern)
        if complete:
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

            # Draw the box on the image
            line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
            line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
            cv.polylines(undistorted_img, [np.int32(line_lower)], True, (255, 0, 0), 2)
            cv.polylines(undistorted_img, [np.int32(line_upper)], True, (0, 0, 255), 2)

            for b, t in zip(line_lower, line_upper):
                cv.line(undistorted_img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

            # Print the camera position
            R, _ = cv.Rodrigues(rvec)
            p = (-R.T @ tvec).flatten()
            info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
            cv.putText(undistorted_img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        # Show the image
        cv.imshow('AR Visualization', undistorted_img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    video.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    # Parameters
    video_file = '../data/chessboard.avi'
    board_pattern = (6, 4)
    board_cellsize = 30  # Arbitrary value

    # Calibrate camera
    K, dist_coeff = calibrate_camera(video_file, board_pattern, board_cellsize)

    # Draw AR
    draw_ar(video_file, K, dist_coeff, board_pattern, board_cellsize)




