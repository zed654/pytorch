import cv2
import numpy as np
import os

# 파일의 디렉토리 경로를 얻기
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 아르코 마커 사전 생성
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# 카메라 캘리브레이션을 위한 아르코 마커의 실제 크기 (미터 단위)
marker_length = 0.05  # 5cm

# 여러 이미지 경로 설정
image_dir = os.path.join(current_dir, "sample_img/calibration")
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# 3D 점과 2D 점 준비
obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane

# 각 이미지에 대해 마커 탐지 및 점 쌍 수집
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 불러올 수 없습니다: {image_file}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # 각 마커에 대해 3D-2D 점 쌍 생성
        for i in range(len(ids)):
            # 3D 점 (마커의 네 모서리)
            objp = np.array([
                [-marker_length/2, marker_length/2, 0],
                [marker_length/2, marker_length/2, 0],
                [marker_length/2, -marker_length/2, 0],
                [-marker_length/2, -marker_length/2, 0]
            ], dtype=np.float32)
            obj_points.append(objp)
            
            # 2D 점 (검출된 마커의 코너)
            img_points.append(corners[i][0])

# 카메라 캘리브레이션 수행
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

if ret:
    print("카메라 캘리브레이션 성공!")
    print("\n카메라 행렬 (K):\n", camera_matrix)
    print("\n왜곡 계수 (P):\n", dist_coeffs)
    
    # 캘리브레이션 결과 저장
    np.save(current_dir + '/result/calibration/camera_matrix.npy', camera_matrix)
    np.save(current_dir + '/result/calibration/dist_coeffs.npy', dist_coeffs)
else:
    print("카메라 캘리브레이션 실패")
