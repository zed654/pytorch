import cv2
import numpy as np
import os

# 파일의 디렉토리 경로를 얻기
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def undistort_image(image_path, camera_matrix, dist_coeffs):
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    # 이미지 크기
    h, w = image.shape[:2]
    
    # 새로운 카메라 행렬 계산
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # 왜곡 보정
    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    # ROI로 이미지 자르기
    x, y, w, h = roi
    undistorted_image = undistorted_image[y:y+h, x:x+w]
    
    return undistorted_image

# 캘리브레이션 결과값을 가상으로 넣어서 왜곡 보정 테스트
def undistort_with_example_coeffs():    
    
    # 샘플 값
    fx = 800
    fy = 800
    cx = 320
    cy = 240

    # 왜곡계수 k(방사형; radition distortion), p(접선형; tangential distortion) 는 마커보드(아르코마커, ...)통해 계산해야함.
    # Radial distortion coefficients
    k1 = -0.2
    k2 = 0.1
    k3 = 0
    # Tangential distortion coefficients
    p1 = 0
    p2 = 0
    
    # 카메라 행렬
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)
    
    # 왜곡 계수
    dist_coeffs = np.array([k1, k2, p1, p2, k3])

    labels_path = os.path.join(current_dir, "sample_img/cat_mari.jpg")
    undistorted_image = undistort_image(labels_path, camera_matrix, dist_coeffs)
    cv2.imwrite(os.path.join(current_dir, "result/undistorted_cat_mari.jpg"), undistorted_image)
    
# calibration.py 실행 후 나온 결과값인 camera_matrix.npy와 dist_coeffs.npy 파일을 불러옴.
def undistort_with_calibration():
    try:
        # 캘리브레이션 결과 불러오기
        camera_matrix = np.load(current_dir + '/result/calibration/camera_matrix.npy')
        dist_coeffs = np.load(current_dir + '/result/calibration/dist_coeffs.npy')
        
        labels_path = os.path.join(current_dir, "sample_img/cat_mari.jpg")
        undistorted_image = undistort_image(labels_path, camera_matrix, dist_coeffs)
        cv2.imwrite(os.path.join(current_dir, "result/undistorted_cat_mari.jpg"), undistorted_image)
        
    except FileNotFoundError:
        print("캘리브레이션 파일을 찾을 수 없습니다. calibration.py를 먼저 실행해주세요.")

if __name__ == "__main__":
     # Example usage
    # undistort_with_example_coeffs()
    undistort_with_calibration()
    
    print("왜곡 보정 완료")
    

