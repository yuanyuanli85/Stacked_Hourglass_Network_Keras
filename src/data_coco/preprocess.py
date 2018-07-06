import cv2
import numpy as np

from src.config.config import DataConfig


def get_crop_area(bbox, imgwidth, imgheight, cfg=DataConfig()):

    scale = cfg.CROP_SCALE
    aspect_ratio = cfg.ASPECT_RATIO

    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox_center_x, bbox_center_y = bbox[0] + bbox[2]//2 , bbox[1] + bbox[3]//2

    if h/w > aspect_ratio:
        w = int(h/aspect_ratio)
    else:
        h = int(w*aspect_ratio)

    top_x = max(0, bbox_center_x - w * scale//2 )
    top_y = max(0, bbox_center_y - h * scale//2 )
    bottom_x = min(imgwidth, bbox_center_x + w * scale//2)
    bottom_y = min(imgheight, bbox_center_y + h * scale//2)

    crop_box = [top_x, top_y, bottom_x-top_x, bottom_y - top_y]
    return np.array(crop_box).astype(np.int)


def crop_image(kp_anno, cfg=DataConfig()):
    cvmat = cv2.imread(kp_anno['filename'])
    bbox =  kp_anno['bbox']
    cropbox = get_crop_area(bbox, kp_anno['width'], kp_anno['height'], cfg)
    cropmat = cvmat[cropbox[1]:cropbox[1]+cropbox[3],
                    cropbox[0]:cropbox[0]+cropbox[2], :]

    # resize crop mat to target
    x_scale =  cfg.IMAGE_WIDTH*1.0 / cropmat.shape[1]
    y_scale =  cfg.IMAGE_HEIGHT*1.0 / cropmat.shape[0]
    resized_mat = cv2.resize(cropmat,(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))

    mkpoints = np.copy(kp_anno['keypoints']).astype(np.float)
    mkpoints[:,0] -= cropbox[0]
    mkpoints[:,0] *= x_scale
    mkpoints[:,1] -= cropbox[1]
    mkpoints[:,1] *= y_scale

    return resized_mat, mkpoints.astype(np.int)

def normalize_image(cvmat, cfg=DataConfig()):
    cvmat = cvmat - cfg.COCO_CHANNEL_MEAN
    cvmat = cvmat / 255.0
    return cvmat

def rotate_image(cvmat, input_Kpoints, angle, cfg=DataConfig()):
    target_width, target_height = cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT
    center = (cvmat.shape[1]//2,  cvmat.shape[0]//2)

    rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotCvmat = cv2.warpAffine(cvmat, rotMat, (target_width, target_height))

    kpoints = np.copy(input_Kpoints)
    kpoints = kpoints.astype(np.float)

    rotkps = np.zeros(kpoints.shape, dtype=np.float)
    for i in range(kpoints.shape[0]):
        x, y = kpoints[i, 0], kpoints[i, 1]
        coor = np.array([x, y])
        if x >= 0 and y >= 0:
            R = rotMat[:, : 2]
            W = np.array([rotMat[0][2], rotMat[1][2]])
            coor = np.dot(R, coor) + W
            rotkps[i, :2] = coor
            rotkps[i, 2] = kpoints[i, 2]

    return rotCvmat, rotkps.astype(np.int)



def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def generate_gtmap(joints, sigma, outres):
    npart = joints.shape[0]
    gtmap = np.zeros(shape=(outres[0], outres[1], npart), dtype=float)
    for i in range(npart):
        visibility = joints[i,2]
        if visibility > 0:
            gtmap[:, :, i] = draw_labelmap(gtmap[:,:,i], joints[i,:], sigma)
    return gtmap


def flip_image(image, keypoints, pairlst):

    h, w, c = image.shape

    flipimg = cv2.flip(image, flipCode=1)

    mkp = np.copy(keypoints)
    mkp[:,0] = w - mkp[:, 0]

    for i, j in pairlst:
        temp = np.copy(mkp[i, :])
        mkp[i, :] = mkp[j, :]
        mkp[j, :] = temp

    return flipimg, mkp.astype(np.int)

