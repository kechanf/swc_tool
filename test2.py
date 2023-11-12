import cv2
import numpy as np
import random
def register_images(image1, image2, keypoints1, keypoints2, matches):
    # 提取关键点的坐标
    pts1 = [keypoints1[match.queryIdx].pt for match in matches]
    pts2 = [keypoints2[match.trainIdx].pt for match in matches]

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    # 计算变换矩阵
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    # 对image2进行变换
    registered_image = cv2.warpPerspective(image2, H, (image1.shape[1], image1.shape[0]))

    return registered_image

def main():
    # 读取两幅图像
    imagefile1 = r'C:\Users\12626\Documents\WeChat Files\wxid_8t1h38c09g6922\FileStorage\File\2023-10\1.png'
    imagefile2 = r'C:\Users\12626\Documents\WeChat Files\wxid_8t1h38c09g6922\FileStorage\File\2023-10\2.png'
    image1 = cv2.imread(imagefile1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(imagefile2, cv2.IMREAD_GRAYSCALE)

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点和计算描述子
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 筛选匹配的特征点
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    # good_matches = random.sample(good_matches, int(0.5*len(good_matches)))



    # 进行图像配准
    registered_image = register_images(image1, image2, keypoints1, keypoints2, good_matches)
    target_width = int(registered_image.shape[1] * 0.5)
    target_height = int(registered_image.shape[0] * 0.5)
    # 使用cv2.resize() 缩小图像
    smaller_img = cv2.resize(registered_image, (target_width, target_height))

    # 显示结果
    cv2.imshow('Registered Image', smaller_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
