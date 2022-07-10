import numpy as np


DATA_PATH = './sample_image_folder'

file_num = 170
data = np.load( DATA_PATH + ('/skeleton_npy/left/left_' + str(file_num) + '.npy') )
print(type(data))
print(data.shape)

# function 1.
def BodyHeight(pose_data):
    head_x = (pose_data[1, 0] + pose_data[2, 0] + pose_data[3, 0] + pose_data[4, 0] + pose_data[5, 0] + pose_data[6, 0]) / 6

    left_foot_x = (pose_data[27, 0] + pose_data[29, 0] + pose_data[31, 0]) / 3
    right_foot_x = (pose_data[28, 0] + pose_data[30, 0] + pose_data[32, 0]) / 3

    head_y = (pose_data[1, 1] + pose_data[2, 1] + pose_data[3, 1] + pose_data[4, 1] + pose_data[5, 1] + pose_data[6, 1]) / 6

    left_foot_y = (pose_data[27, 1] + pose_data[29, 1] + pose_data[31, 1]) / 3
    right_foot_y = (pose_data[28, 1] + pose_data[30, 1] + pose_data[32, 1]) / 3

    head_z = (pose_data[1, 2] + pose_data[2, 2] + pose_data[3, 2] + pose_data[4, 2] + pose_data[5, 2] + pose_data[6, 2]) / 6

    left_foot_z = (pose_data[27, 2] + pose_data[29, 2] + pose_data[31, 2]) / 3
    right_foot_z = (pose_data[28, 2] + pose_data[30, 2] + pose_data[32, 2]) / 3

    # make result.
    body_height_x = abs(head_x - max(left_foot_x, right_foot_x))
    body_height_y = abs(head_y - max(left_foot_y, right_foot_y))
    body_height_z = abs(head_z - max(left_foot_z, right_foot_z))

    print("body_height_x : ", body_height_x)
    print("body_height_y : ", body_height_y)
    print("body_height_z : ", body_height_z)

    return body_height_x, body_height_y, body_height_z


def Head_Neck_Distance(pose_data):

    head_x = (pose_data[1, 0] + pose_data[2, 0] + pose_data[3, 0] + pose_data[4, 0] + pose_data[5, 0] + pose_data[6, 0]) / 6
    head_y = (pose_data[1, 1] + pose_data[2, 1] + pose_data[3, 1] + pose_data[4, 1] + pose_data[5, 1] + pose_data[6, 1]) / 6
    head_z = (pose_data[1, 2] + pose_data[2, 2] + pose_data[3, 2] + pose_data[4, 2] + pose_data[5, 2] + pose_data[6, 2]) / 6

    neck_x = (pose_data[11, 0] + pose_data[12, 0]) / 2
    neck_y = (pose_data[11, 1] + pose_data[12, 1]) / 2
    neck_z = (pose_data[11, 2] + pose_data[12, 2]) / 2

    dist_x = abs(head_x - neck_x)
    dist_y = abs(head_y - neck_y)
    dist_z = abs(head_z - neck_z)

    return dist_x, dist_y, dist_z


#body_height_x, body_height_y, body_height_z = BodyHeight(data)


# function 2.
def LeftHand_Head(pose_data):
    left_hand_x = (pose_data[15, 0] + pose_data[17, 0] + pose_data[19, 0] + pose_data[21, 0]) / 4
    head_x = (pose_data[1, 0] + pose_data[2, 0] + pose_data[3, 0] + pose_data[4, 0] + pose_data[5, 0] + pose_data[6, 0]) / 6

    left_hand_y = (pose_data[15, 1] + pose_data[17, 1] + pose_data[19, 1] + pose_data[21, 1]) / 4
    head_y = (pose_data[1, 1] + pose_data[2, 1] + pose_data[3, 1] + pose_data[4, 1] + pose_data[5, 1] + pose_data[6, 1]) / 6

    left_hand_z = (pose_data[15, 2] + pose_data[17, 2] + pose_data[19, 2] + pose_data[21, 2]) / 4
    head_z = (pose_data[1, 2] + pose_data[2, 2] + pose_data[3, 2] + pose_data[4, 2] + pose_data[5, 2] + pose_data[6, 2]) / 6

    # make result.
    body_height_x, body_height_y, body_height_z = BodyHeight(data)

    is_close_x = (abs(head_x - left_hand_x) < body_height_x * 0.15)
    is_close_y = (abs(head_y - left_hand_y) < body_height_y * 0.15)
    is_close_z = (abs(head_z - left_hand_z) < body_height_z * 0.15)

    return is_close_x, is_close_y, is_close_z


# function 3.
def RightHand_Head(pose_data):
    right_hand_x = (pose_data[16, 0] + pose_data[18, 0] + pose_data[20, 0] + pose_data[22, 0]) / 4
    head_x = (pose_data[1, 0] + pose_data[2, 0] + pose_data[3, 0] + pose_data[4, 0] + pose_data[5, 0] + pose_data[6, 0]) / 6

    right_hand_y = (pose_data[16, 1] + pose_data[18, 1] + pose_data[20, 1] + pose_data[22, 1]) / 4
    head_y = (pose_data[1, 1] + pose_data[2, 1] + pose_data[3, 1] + pose_data[4, 1] + pose_data[5, 1] + pose_data[6, 1]) / 6

    right_hand_z = (pose_data[16, 2] + pose_data[18, 2] + pose_data[20, 2] + pose_data[22, 2]) / 4
    head_z = (pose_data[1, 2] + pose_data[2, 2] + pose_data[3, 2] + pose_data[4, 2] + pose_data[5, 2] + pose_data[6, 2]) / 6

    # make result.
    body_height_x, body_height_y, body_height_z = BodyHeight(data)

    is_close_x = (abs(head_x - right_hand_x) < body_height_x * 0.15)
    is_close_y = (abs(head_y - right_hand_y) < body_height_y * 0.15)
    is_close_z = (abs(head_z - right_hand_z) < body_height_z * 0.15)

    return is_close_x, is_close_y, is_close_z


#is_close_x, is_close_y, is_close_z = LeftHand_Head(data)

# print(is_close_x)
# print(is_close_y)
# print(is_close_z)

def Visi_Check_Foot(pose_data):
    left_foot_v = (pose_data[27, 3] + pose_data[29, 3] + pose_data[31, 3]) / 3
    right_foot_v = (pose_data[28, 3] + pose_data[30, 3] + pose_data[32, 3]) / 3

    if left_foot_v < 0.5 and right_foot_v < 0.5:
        return False
    else:
        return True

#Visi_Check_Foot(data)



def LeftShoulder_Minus_RightShoulder(pose_data):

    lsmrs_x = pose_data[11, 0] - pose_data[12, 0]
    lsmrs_y = pose_data[11, 1] - pose_data[12, 1]
    lsmrs_z = pose_data[11, 2] - pose_data[12, 2]

    return lsmrs_x, lsmrs_y, lsmrs_z


def Is_Near_Left(pose_data):

    head_x = (pose_data[1, 0] + pose_data[2, 0] + pose_data[3, 0] + pose_data[4, 0] + pose_data[5, 0] + pose_data[6, 0]) / 6
    head_y = (pose_data[1, 1] + pose_data[2, 1] + pose_data[3, 1] + pose_data[4, 1] + pose_data[5, 1] + pose_data[6, 1]) / 6
    head_z = (pose_data[1, 2] + pose_data[2, 2] + pose_data[3, 2] + pose_data[4, 2] + pose_data[5, 2] + pose_data[6, 2]) / 6

    left_shoulder_x = pose_data[11, 0]
    left_shoulder_y = pose_data[11, 1]
    left_shoulder_z = pose_data[11, 2]

    right_shoulder_x = pose_data[12, 0]
    right_shoulder_y = pose_data[12, 1]
    right_shoulder_z = pose_data[12, 2]

    dist_left_head_x = abs(head_x - left_shoulder_x)
    dist_left_head_y = abs(head_y - left_shoulder_y)
    dist_left_head_z = abs(head_z - left_shoulder_z)

    dist_right_head_x = abs(head_x - right_shoulder_x)
    dist_right_head_y = abs(head_y - right_shoulder_y)
    dist_right_head_z = abs(head_z - right_shoulder_z)

    if (dist_left_head_x + dist_left_head_y + dist_left_head_z < 
        dist_right_head_x + dist_right_head_y + dist_right_head_z):
        return True
    else:
        return False



def Is_Near_Elbow(pose_data):

    elbow_x = (pose_data[13, 0] + pose_data[14, 0]) / 2
    elbow_y = (pose_data[13, 1] + pose_data[14, 1]) / 2
    elbow_z = (pose_data[13, 2] + pose_data[14, 2]) / 2

    shoulder_x = (pose_data[11, 0] + pose_data[12, 0]) / 2
    shoulder_y = (pose_data[11, 1] + pose_data[12, 1]) / 2
    shoulder_z = (pose_data[11, 2] + pose_data[12, 2]) / 2

    head_x = (pose_data[1, 0] + pose_data[2, 0] + pose_data[3, 0] + pose_data[4, 0] + pose_data[5, 0] + pose_data[6, 0]) / 6
    head_y = (pose_data[1, 1] + pose_data[2, 1] + pose_data[3, 1] + pose_data[4, 1] + pose_data[5, 1] + pose_data[6, 1]) / 6
    head_z = (pose_data[1, 2] + pose_data[2, 2] + pose_data[3, 2] + pose_data[4, 2] + pose_data[5, 2] + pose_data[6, 2]) / 6

    print(abs(elbow_x - head_x))
    print(abs(shoulder_x - head_x))

    if abs(elbow_x - head_x) < abs(shoulder_x - head_x):
        return True
    else:
        return False




def Make_Inference_Left(data):

    if Visi_Check_Foot(data) is True:        

        is_close_x, is_close_y, is_close_z = LeftHand_Head(data)

        print(" *** Inference Result *** ")
        print("is_close_x : ", is_close_x)
        print("is_close_y : ", is_close_y)
        print("is_close_z : ", is_close_z)

        if is_close_x and is_close_y:
            print("=== The input pose is estimated to be tilted to left! ===")
            return True
        else:
            return False

    else:

        hn_dist_x, hn_dist_y, hn_dist_z = Head_Neck_Distance(data)
        lsmrs_x, lsmrs_y, lsmrs_z = LeftShoulder_Minus_RightShoulder(data)
        
        print(" *** Inference Result *** ")
        print("hn_dist_x, hn_dist_y, hn_dist_z : ", hn_dist_x, hn_dist_y, hn_dist_z)
        print("lsmrs_x, lsmrs_y, lsmrs_z : ", lsmrs_x, lsmrs_y, lsmrs_z)
        
        if lsmrs_x > hn_dist_x * 0.5 and lsmrs_y > hn_dist_y * 0.5:
            print("=== The input pose is estimated to be tilted to left! ===")
            return True
        else:
            if Is_Near_Left(data) is True:
                print("=== The input pose is estimated to be tilted to left! ===")
                return True
            else:
                return False

#Make_Inference_Left(data)



def Make_Inference_Right(data):

    if Visi_Check_Foot(data) is True:        

        is_close_x, is_close_y, is_close_z = RightHand_Head(data)

        print(" *** Inference Result *** ")
        print("is_close_x : ", is_close_x)
        print("is_close_y : ", is_close_y)
        print("is_close_z : ", is_close_z)

        if is_close_x and is_close_y:
            print("=== The input pose is estimated to be tilted to right! ===")
            return True
        else:
            return False

    else:

        hn_dist_x, hn_dist_y, hn_dist_z = Head_Neck_Distance(data)
        lsmrs_x, lsmrs_y, lsmrs_z = LeftShoulder_Minus_RightShoulder(data)
        
        print(" *** Inference Result *** ")
        print("hn_dist_x, hn_dist_y, hn_dist_z : ", hn_dist_x, hn_dist_y, hn_dist_z)
        print("lsmrs_x, lsmrs_y, lsmrs_z : ", lsmrs_x, lsmrs_y, lsmrs_z)
        
        if lsmrs_x < hn_dist_x * (-0.5) and lsmrs_y < hn_dist_y * (-0.5):
            print("=== The input pose is estimated to be tilted to right! ===")
            return True
        else:
            if Is_Near_Left(data) is False:
                print("=== The input pose is estimated to be tilted to right! ===")
                return True
            else:
                return False




def Make_Inference_Turtle(data):

    if Is_Near_Elbow(data) is True:
        print("=== The input pose is estimated to be tilted to turtleneck! ===")
        return True

    else:        
        return False




## main ##
# count_yes = 0
# count_no = 0
count_left = 0
count_right = 0
count_turtle = 0
count_good = 0

for file_num in range(410):

    # Load data here.
    #if file_num == 263: continue       # for right.
    data = np.load( DATA_PATH + ('/skeleton_npy/turtleneck/turtleneck_' + str(file_num) + '.npy') )

    print("Now Making Result of", str(file_num) + '.npy', ". . .")

    # Make inference here.
    is_left = Make_Inference_Left(data)
    if is_left is True:
        count_left += 1
        continue
    
    is_right = Make_Inference_Right(data)
    if is_right is True:
        count_right += 1
        continue
        
    is_turtle = Make_Inference_Turtle(data)
    if is_turtle is True:
        count_turtle += 1
        continue

    print("=== The input pose is estimated to be good! ===")
    count_good += 1

print("  --- Inference Result of Turtleneck  ---  ")
print("count_good   : ", count_good)
print("count_left   : ", count_left)
print("count_right  : ", count_right)
print("count_turtle : ", count_turtle)


# print("count_yes : ", count_yes)
# print("count_no  : ", count_no)