import os
import json
import numpy as np

def vecXlist_to_numpy(vecX, name=''):
    localX = []
    for i in range(len(vecX)):
        if len(vecX[i]) == 3:
            x, y, z = vecX[i]['x'],  vecX[i]['y'],  vecX[i]['z']
            if "right" in name:
                localx = [-x, -z, y]
                #localx = [z, y, x]
                pass
            elif "left" in name:
                localx = [x, -z, -y]
                #localx = [-z, y, -x]
            elif "head" in name:
                localx = [z,-x, y]
        
            else:
                localx = [x, y, z]
                
        if len(vecX[i]) == 4:
            x, y, z, w = vecX[i]['x'],  vecX[i]['y'],  vecX[i]['z'],   vecX[i]['w']
            localx = [x, y, z, w]

        localX.append(localx)
        
    result =  np.array(localX)
    return result

def from_txt_to_npy(path, txt_name, save_np, np_name, fps=30, preprocess=True):
    txt_file = open(path + txt_name+".txt")
    jsonfile = json.load(txt_file)

    # parse motion info
    frameCount = jsonfile['frameCount']
    rcontroller_local_pos = jsonfile['rcontroller_local_pos']
    lcontroller_local_pos = jsonfile['lcontroller_local_pos']
    headset_local_pos = jsonfile['headset_local_pos']
    rcontroller_local_rot = jsonfile['rcontroller_local_rot']
    lcontroller_local_rot = jsonfile['lcontroller_local_rot']
    headset_local_rot = jsonfile['headset_local_rot']
    # isXpressed = jsonfile['isXPressed']
    # isYpressed = jsonfile['isYPressed']
    
    rightLpos = vecXlist_to_numpy(rcontroller_local_pos, "rightLpos")
    leftLpos = vecXlist_to_numpy(lcontroller_local_pos, "leftLpos")
    headLpos = vecXlist_to_numpy(headset_local_pos, "headLpos")
    rightLRot = vecXlist_to_numpy(rcontroller_local_rot)
    leftLRot = vecXlist_to_numpy(lcontroller_local_rot)
    headLRot = vecXlist_to_numpy(headset_local_rot)
    # isXpressed = np.array(isXpressed)
    # isYpressed = np.array(isYpressed)
    
    # changejnt_trans to numpy
    print("=========")
    print("frameCount:", frameCount)
    print("rightLpos: ", rightLpos.shape)
    print("leftLpos: ", leftLpos.shape)
    print("headLpos: ", headLpos.shape)
    # print("isXpressed: ", isXpressed.shape)
    # print("isYpressed: ", isYpressed.shape)
    print("=========")
    
    if save_np:
        print("--------------- save Transformation Matrix npy file: ", np_name, "--------------")
        np.save(path + txt_name + "@rlh_localPos", np.concatenate((rightLpos, leftLpos, headLpos), axis=1))
        np.save(path + txt_name + "@rlh_localRot", np.concatenate((rightLRot, leftLRot, headLRot), axis=1))
        # np.save(path + txt_name + "@xy_pressed", np.concatenate((np.expand_dims(isXpressed, axis=1), \
                                                        # np.expand_dims(isYpressed, axis=1)), axis=1))
    return rightLpos, leftLpos, headLpos

if __name__ == "__main__":
    path = os.getcwd()
    txt_name = "/data/unity/joystick_input/joystick0"
    rightLpos, leftLpos, headLpos = \
        from_txt_to_npy(path, txt_name, True, np_name="1016_pickfruits_joystick0", preprocess=True)
    