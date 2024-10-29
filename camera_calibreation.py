import cv2 as cv
import os, shutil
from tqdm import tqdm
from glob import glob
import numpy as np

def select_img_from_video(video_file):
    video = cv.VideoCapture(video_file)
    
    if not video.isOpened():
        assert False, "Video load error"
    
    len_video = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv.CAP_PROP_FPS))
    
    folder_name = './save_img'
    images_save_folder = os.path.join(folder_name)
    if os.path.exists(images_save_folder):
        shutil.rmtree(images_save_folder)
    os.makedirs(images_save_folder)
    
    count = 0; success = True
    with tqdm(total = len_video + 1) as pbar:
        while success:
            success, image = video.read()
            frame_idx = int(video.get(1))
            if frame_idx % fps == 0:
                save_idx = str(count + 1)
                save_image_path = os.path.join(images_save_folder, f"frame_{save_idx}.jpg")
                cv.imwrite(save_image_path, image)
                count += 1
            pbar.update(1)
    video.release()
    
    return images_save_folder

def camera_calibrate(img_path, checkerboard):
    img_files = glob(img_path + '/*.jpg')
    t_img = []
    
    folder_name = './chessboard_corner'
    images_save_folder = os.path.join(folder_name)
    if os.path.exists(images_save_folder):
        shutil.rmtree(images_save_folder)
    os.makedirs(images_save_folder)
    
    for fname in img_files:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, checkerboard,
                                             cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            base_name = os.path.basename(fname)
            t_img.append(fname)
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)
            img = cv.drawChessboardCorners(img, checkerboard, corners2, ret)
            cv.imwrite(f"./chessboard_corner/corner_{base_name}.jpg", img)
            
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs, t_img

def calculate_rms_per_image(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        errors.append(error)
    return errors

def save_undistorted_images(t_img, mtx, dist):
    undistorted_folder = './save_img/undistorted_images'
    os.makedirs(undistorted_folder, exist_ok=True)
    
    for img_file in t_img:
        img = cv.imread(img_file)
        undistorted_img = cv.undistort(img, mtx, dist, None, mtx)
        
        base_name = os.path.basename(img_file)
        undistorted_img_path = os.path.join(undistorted_folder, f"undistorted_{base_name}")
        cv.imwrite(undistorted_img_path, undistorted_img)

def visualize_camera_pose(img_path, rvecs, tvecs, mtx, dist, axis_length=3.0):
    img_files = glob(img_path + '/*.jpg')
    for idx, fname in enumerate(img_files):
        img = cv.imread(fname)
        axis_points = np.float32([[0,0,0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, -axis_length]]).reshape(-1, 3)
        
        img_points, _ = cv.projectPoints(axis_points, rvecs[idx], tvecs[idx], mtx, dist)
        
        points = [tuple(map(int, point.ravel())) for point in img_points]
        origin = points[0]
        img = cv.line(img, origin, points[1], (0,0,255), 5)     # X axis in red
        img = cv.line(img, origin, points[2], (0,255,0), 5)     # Y axis in green
        img = cv.line(img, origin, points[3], (255,0,0), 5)     # Z axis in blue
        
        font = cv.FONT_HERSHEY_SIMPLEX
        img = cv.putText(img, 'X', points[1], font, 1, (0,0,255), 2)
        img = cv.putText(img, 'Y', points[2], font, 1, (0,255,0), 2)
        img = cv.putText(img, 'Z', points[3], font, 1, (255,0,0), 2)
        
        cv.imwrite(f"./save_img/pose_{idx+1}.jpg", img)

video_file = './my_chessboard.mp4'
checkerboard = (7, 10)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []

objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
prev_img_shape = None

img_path = select_img_from_video(video_file)
objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs, t_img = camera_calibrate(img_path, checkerboard)
rms_errors = calculate_rms_per_image(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
# visualize_camera_pose(img_path, rvecs, tvecs, mtx, dist, axis_length=3.0)

for idx, error in enumerate(rms_errors):
    print(f"Image {t_img[idx]} RMS Error: {error}") # rms 오차

save_undistorted_images(t_img, mtx, dist)

print(f"Number of images: {len(rms_errors)}")   # 사용 이미지 개수
print(f"Average RMS Error: {np.mean(rms_errors)}")

print("Camera matrix : \n", mtx) # 카메라 내부 파라미터
# print("Distortion coefficients : \n", dist)
# print("Rotation vectors : \n", rvecs)
# print("Translation vectors : \n", tvecs)