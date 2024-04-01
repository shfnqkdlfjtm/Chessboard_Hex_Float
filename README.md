# Chessboard_Hex_Float
Floating hexahedron on chessboard using Camera Calibration

## 프로그램 및 기능 설명
이 코드는 카메라를 보정하고 보정된 카메라를 사용하여 화면에 가상의 3D 객체를 그리는 간단한 증강 현실(AR) 시스템을 구현한다. 이를 위해 먼저 체스보드 패턴을 사용하여 카메라를 보정합니다. 보정된 카메라를 사용하여 카메라의 위치와 방향을 추정하고, 추정된 위치와 방향을 기반으로 가상의 3D 객체를 그려 화면에 표시한다.
calibrate_camera(video_file, board_pattern, board_cellsize, dist_coeff=None): 이 함수는 체스보드 패턴을 사용하여 카메라를 보정하다. 먼저 주어진 비디오 파일에서 프레임을 읽어와 각 프레임에서 체스보드 코너를 검출한다. 검출된 코너들을 사용하여 카메라 보정을 수행하고, 카메라의 내부 매개변수(K 행렬)와 왜곡 계수(dist_coeff)를 반환한다.

draw_ar(video_file, K, dist_coeff, board_pattern, board_cellsize): 이 함수는 보정된 카메라를 사용하여 가상의 3D 객체를 그려 화면에 표시한다. 함수는 주어진 비디오 파일에서 프레임을 읽어오고, 각 프레임에서 체스보드 코너를 검출하여 카메라의 위치와 방향을 추정한다. 추정된 카메라의 위치와 방향을 기반으로 3D 객체를 그려 화면에 표시하고, 카메라의 위치 정보를 화면에 표시한다.

메인 코드는 위의 두 함수를 호출하여 전체 프로그램을 실행한다. 먼저 calibrate_camera() 함수를 사용하여 카메라를 보정하고, 그 결과를 이용하여 draw_ar() 함수를 호출하여 가상의 3D 객체를 그려 화면에 표시한다.

## Camera Calibration 코드
    import cv2 as cv
    import numpy as np
    
    def calibrate_camera(video_file, board_pattern, board_cellsize, dist_coeff=None):
        # Open a video
        video = cv.VideoCapture(video_file)
    
        # Prepare 3D points on a chessboard
        obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])
    
        # Collect chessboard points from video frames
        img_points = []
        while True:
            # Read an image from the video
            valid, img = video.read()
            if not valid:
                break
    
            # Find chessboard corners
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            complete, pts = cv.findChessboardCorners(gray, board_pattern)
            if complete:
                img_points.append(pts)
    
        assert len(img_points) > 0, 'No complete chessboard points found in the video frames.'
    
        # Calibrate the camera
        ret, K, dist_coeff, _, _ = cv.calibrateCamera([obj_points] * len(img_points), img_points, gray.shape[::-1], None, None, flags=cv.CALIB_USE_INTRINSIC_GUESS)
    
        # Release the video capture
        video.release()
    
        return K, dist_coeff
    
    def draw_ar(video_file, K, dist_coeff, board_pattern, board_cellsize):
        # Open a video
        video = cv.VideoCapture(video_file)
    
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
    
            # Estimate the camera pose
            complete, img_points = cv.findChessboardCorners(img, board_pattern)
            if complete:
                ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)
    
                # Draw the box on the image
                line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
                line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
                cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
                cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)
    
                for b, t in zip(line_lower, line_upper):
                    cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
    
                # Print the camera position
                R, _ = cv.Rodrigues(rvec)
                p = (-R.T @ tvec).flatten()
                info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
                cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    
            # Show the image
            cv.imshow('AR Visualization', img)
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
    
## Camera Calibration 결과


## 렌즈 왜곡 보정 추가 코드
    import cv2 as cv
    import numpy as np
    from distortion_correction import calibrate_camera, undistort_image
    
    def draw_ar(video_file, K, dist_coeff, board_pattern, board_cellsize):
        # Open a video
        video = cv.VideoCapture(video_file)
    
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


## 렌즈 왜곡 보정 결과
