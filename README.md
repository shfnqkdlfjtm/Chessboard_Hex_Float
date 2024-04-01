# Chessboard_Hex_Float
Floating hexahedron on chessboard using Camera Calibration

## 프로그램 및 기능 설명
이 코드는 카메라를 보정하고 보정된 카메라를 사용하여 화면에 가상의 3D 객체를 그리는 간단한 증강 현실(AR) 시스템을 구현한다. 이를 위해 먼저 체스보드 패턴을 사용하여 카메라를 보정합니다. 보정된 카메라를 사용하여 카메라의 위치와 방향을 추정하고, 추정된 위치와 방향을 기반으로 가상의 3D 객체를 그려 화면에 표시한다.
calibrate_camera(video_file, board_pattern, board_cellsize, dist_coeff=None): 이 함수는 체스보드 패턴을 사용하여 카메라를 보정하다. 먼저 주어진 비디오 파일에서 프레임을 읽어와 각 프레임에서 체스보드 코너를 검출한다. 검출된 코너들을 사용하여 카메라 보정을 수행하고, 카메라의 내부 매개변수(K 행렬)와 왜곡 계수(dist_coeff)를 반환한다.

draw_ar(video_file, K, dist_coeff, board_pattern, board_cellsize): 이 함수는 보정된 카메라를 사용하여 가상의 3D 객체를 그려 화면에 표시한다. 함수는 주어진 비디오 파일에서 프레임을 읽어오고, 각 프레임에서 체스보드 코너를 검출하여 카메라의 위치와 방향을 추정한다. 추정된 카메라의 위치와 방향을 기반으로 3D 객체를 그려 화면에 표시하고, 카메라의 위치 정보를 화면에 표시한다.

메인 코드는 위의 두 함수를 호출하여 전체 프로그램을 실행한다. 먼저 calibrate_camera() 함수를 사용하여 카메라를 보정하고, 그 결과를 이용하여 draw_ar() 함수를 호출하여 가상의 3D 객체를 그려 화면에 표시한다.

## Camera Calibration 코드
    import numpy as np
    import cv2 as cv
    
    # The given video and calibration data
    video_file = r"C:\Users\samsung\Downloads\chessboard.avi"
    K = np.array([[432.7390364738057, 0, 476.0614994349778],
                  [0, 431.2395555913084, 288.7602152621297],
                  [0, 0, 1]])
    dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194])
    board_pattern = (10, 7)
    board_cellsize = 0.025
    board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
    
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), 'Cannot read the given input, ' + video_file
    
    # Prepare a 3D box for simple AR
    box_lower = board_cellsize * np.array([[4, 2,  0], [5, 2,  0], [5, 4,  0], [4, 4,  0]])
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
        success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
        if success:
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)
    
            # Draw the box on the image
            line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
            line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
            cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)
            for b, t in zip(line_lower, line_upper):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
    
            # Print the camera position
            R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
            p = (-R.T @ tvec).flatten()
            info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
            cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    
        # Show the image and process the key event
        cv.imshow('Pose Estimation (Chessboard)', img)
        key = cv.waitKey(10)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break

    video.release()
    cv.destroyAllWindows()
    
## Camera Calibration 결과
![camera_calibration_screenshot](https://github.com/shfnqkdlfjtm/cv.CaptureCraft/assets/144716487/795238e5-6793-4d14-b506-9d653db7156e)
특별한 보정 없이도 체스보드 위에 육면체가 그려진다는 것을 확인할 수 있다.

## 렌즈 왜곡 보정 추가 코드
    import numpy as np
    import cv2 as cv
    
    # The given video and calibration data
    video_file = r"C:\Users\samsung\Downloads\chessboard.avi"
    K = np.array([[432.7390364738057, 0, 476.0614994349778],
                  [0, 431.2395555913084, 288.7602152621297],
                  [0, 0, 1]])
    dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 0.0001149909868437517, -0.01803978785585194])
    board_pattern = (10, 7)
    board_cellsize = 0.025
    board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
    
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), 'Cannot read the given input, ' + video_file
    
    # Prepare a 3D box for simple AR
    box_lower = board_cellsize * np.array([[4, 2,  0], [5, 2,  0], [5, 4,  0], [4, 4,  0]])
    box_upper = board_cellsize * np.array([[4, 2, -1], [5, 2, -1], [5, 4, -1], [4, 4, -1]])
    
    # Prepare 3D points on a chessboard
    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])
    
    # Run pose estimation
    while True:
        # Read an image from the video
        valid, img = video.read()
        if not valid:
           break
    
        # Apply lens distortion correction
        img = cv.undistort(img, K, dist_coeff)
    
        # Estimate the camera pose
        success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
        if success:
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)
    
            # Draw the box on the image
            line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
            line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
            cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)
            cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 2)
            for b, t in zip(line_lower, line_upper):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)
    
            # Print the camera position
            R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
            p = (-R.T @ tvec).flatten()
            info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
            cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
    
        # Show the image and process the key event
        cv.imshow('Pose Estimation (Chessboard)', img)
        key = cv.waitKey(10)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break
    
    video.release()
    cv.destroyAllWindows()
    
    
            cv.destroyAllWindows()
    
    
## 렌즈 왜곡 보정 결과
![camera_correction_screenshot](https://github.com/shfnqkdlfjtm/cv.CaptureCraft/assets/144716487/ee9efe6b-a0db-4ecc-97fb-67cc6c7033d2)
오히려 왜곡이 더 심해진 것을 확인할 수 있다.
