
import numpy as np
import cv2
from cv2 import aruco
import math


def detect_ArUco_details(image):
    markerSize = 4
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_250')
    Dict = aruco.getPredefinedDictionary(key)
    params = aruco.DetectorParameters()
    corners, ids, rejected = aruco.detectMarkers(imgGray, Dict, parameters=params)
    sa = []
    
    for i in range(len(ids)):
        ca = np.array(corners[i])
        cb = np.array(ca[0])
        sa.append(cb)
    fi= tuple(sa)
    centers_angles = []
        
    if ids is not None:
        for i in range(len(ids)):
            centers = []
            center = np.mean(corners[i][0], axis=0).astype(int)
            centres = center.tolist()
            centers.append(centres)
            dx = corners[i][0][1][0] - corners[i][0][0][0]
            dy = corners[i][0][1][1] - corners[i][0][0][1]
            angle_rad = np.arctan2(dy, dx)
            centers.append(np.degrees(angle_rad))
            centers_angles.append(centers)
    ArUco_details_dict = {id[0]:center for id,center in zip(ids,centers_angles)}
    ArUco_corners = {id[0]: corner for id, corner in zip(ids, fi)}
    
    
    return ArUco_details_dict, ArUco_corners 


def mark_ArUco_image(image,ArUco_details_dict, ArUco_corners):

    for ids, details in ArUco_details_dict.items():
        center = details[0]
        cv2.circle(image, center, 5, (0,0,255), -1)

        corner = ArUco_corners[int(ids)]
        cv2.circle(image, (int(corner[0][0]), int(corner[0][1])), 5, (50, 50, 50), -1)
        cv2.circle(image, (int(corner[1][0]), int(corner[1][1])), 5, (0, 255, 0), -1)
        cv2.circle(image, (int(corner[2][0]), int(corner[2][1])), 5, (128, 0, 255), -1)
        cv2.circle(image, (int(corner[3][0]), int(corner[3][1])), 5, (25, 255, 255), -1)

        tl_tr_center_x = int((corner[0][0] + corner[1][0]) / 2)
        tl_tr_center_y = int((corner[0][1] + corner[1][1]) / 2) 

        cv2.line(image,center,(tl_tr_center_x, tl_tr_center_y),(255,0,0),5)
        display_offset = int(math.sqrt((tl_tr_center_x - center[0])**2+(tl_tr_center_y - center[1])**2))
        cv2.putText(image,str(ids),(center[0]+int(display_offset/2),center[1]),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        angle = details[1]
        cv2.putText(image,str(angle),(center[0]-display_offset,center[1]),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return image

if __name__ == "__main__":

    # path directory of images in test_images folder
    img_dir_path = "public_test_cases/"

    marker = 'aruco'

    for file_num in range(0,2):
        img_file_path = img_dir_path +  marker + '_' + str(file_num) + '.png'

        # read image using opencv
        img = cv2.imread(img_file_path)

        print('\n============================================')
        print('\nFor '+ marker  +  str(file_num) + '.png')
   
        ArUco_details_dict, ArUco_corners = detect_ArUco_details(img)
        print("Detected details of ArUco: " , ArUco_details_dict )

        #displaying the marked image
        img = mark_ArUco_image(img, ArUco_details_dict, ArUco_corners) 
        cv2.imshow("Marked Image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
