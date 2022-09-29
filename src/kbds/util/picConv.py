import math


def crop_object(image, obj_meta):
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)

    crop_img = image[top:top+height, left:left+width]
	
    return crop_img

def crop_object(frame, coor):
    top = int(coor[0])
    left = int(coor[1])
    width = int(coor[2])
    height = int(coor[3])

    crop_img = frame[top:top+height, left:left+width]
	
    return crop_img
