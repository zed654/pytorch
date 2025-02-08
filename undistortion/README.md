샘플 이미지로 캘리브레이션 후 왜곡 보정 테스트

1. calibration.py
    - 샘플 이미지로 캘리브레이션 통해 Intrinsic 파라미터 및 왜곡계수(k, p) 추정
    - 결과는 result에 저장됨

2. undistortion.py
    - 캘리브레이션 결과값을 가상으로 넣어서 왜곡 보정 테스트
        - undistort_with_example_coeffs()
            - 왜곡계수(k, p) 가상으로 넣어서 왜곡 보정 테스트
        - undistort_with_calibration()
            - calibration.py 실행 후 나온 결과값인 camera_matrix.npy와 dist_coeffs.npy 파일을 불러옴.
    - 결과는 result 폴더에 저장됨.

