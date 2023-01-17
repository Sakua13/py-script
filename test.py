import cv2
import mediapipe as mp
import math
import time
import csv
import os

# # 2頂点の距離の計算
# def calcDistance(p0, p1):
#   a1 = p1.x-p0.x
#   a2 = p1.y-p0.y
#   return math.sqrt(a1*a1 + a2*a2)

# # 3頂点の角度の計算
# def calcAngle(p0, p1, p2):
#   a1 = p1.x-p0.x
#   a2 = p1.y-p0.y
#   b1 = p2.x-p1.x
#   b2 = p2.y-p1.y
#   angle = math.acos( (a1*b1 + a2*b2) / math.sqrt((a1*a1 + a2*a2)*(b1*b1 + b2*b2)) ) * 180/math.pi
#   return angle

# # 指の角度の合計の計算
# def cancFingerAngle(p0, p1, p2, p3, p4):
#   result = 0
#   result += calcAngle(p0, p1, p2)
#   result += calcAngle(p1, p2, p3)
#   result += calcAngle(p2, p3, p4)
#   return result

# # 指ポーズの検出
# def detectFingerPose(hand_landmarks):
#   # 指のオープン・クローズ
#   thumbIsOpen = cancFingerAngle(hand_landmarks[0], hand_landmarks[1], hand_landmarks[2], hand_landmarks[3], hand_landmarks[4]) < 70
#   print("a")
#   firstFingerIsOpen = cancFingerAngle(hand_landmarks[0], hand_landmarks[5], hand_landmarks[6], hand_landmarks[7], hand_landmarks[8]) < 100
#   secondFingerIsOpen = cancFingerAngle(hand_landmarks[0], hand_landmarks[9], hand_landmarks[10], hand_landmarks[11], hand_landmarks[12]) < 100
#   thirdFingerIsOpen = cancFingerAngle(hand_landmarks[0], hand_landmarks[13], hand_landmarks[14], hand_landmarks[15], hand_landmarks[16]) < 100
#   fourthFingerIsOpen = cancFingerAngle(hand_landmarks[0], hand_landmarks[17], hand_landmarks[18], hand_landmarks[19], hand_landmarks[20]) < 100

#   # ジェスチャー
#   if (calcDistance(hand_landmarks[4], hand_landmarks[8]) < 0.1 and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen):
#     # return "OK"
#     print("OK")
#   elif (calcDistance(hand_landmarks[4], hand_landmarks[12]) < 0.1 and calcDistance(hand_landmarks[4], hand_landmarks[16]) < 0.1 and firstFingerIsOpen and fourthFingerIsOpen):
#     # return "キツネ"
#     print("狐")
#   elif (thumbIsOpen and not firstFingerIsOpen and (not secondFingerIsOpen) and (not thirdFingerIsOpen) and (not fourthFingerIsOpen)):
#     # return "いいね"
#     print("GOOD")
#   elif (thumbIsOpen and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen):
#     # return "５"
#     print("5")
#   elif ((not thumbIsOpen) and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen):
#     # return "４"
#     print("4")
#   elif ((not thumbIsOpen) and firstFingerIsOpen and secondFingerIsOpen and thirdFingerIsOpen and (not fourthFingerIsOpen)):
#     # return "３"
#     print("3")
#   elif ((not thumbIsOpen) and firstFingerIsOpen and secondFingerIsOpen and (not thirdFingerIsOpen) and (not fourthFingerIsOpen)):
#     # return "２"
#     print("2")
#   elif ((not thumbIsOpen) and firstFingerIsOpen and (not secondFingerIsOpen) and (not thirdFingerIsOpen) and (not fourthFingerIsOpen)):
#     # return "１"
#     print("1")
#   else:
#     # return "１"
#     print("1")



def get_status(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        thumb = hand_landmarks[4]           # 親指
        index_finger = hand_landmarks[8]    # 人差し指
        middle_finger = hand_landmarks[12]  # 中指
        ring_finger = hand_landmarks[16]    # 薬指
        pinky_finger = hand_landmarks[20]   # 小指
        # 親指が他の指より上にある場合（上部からのパーセンテージが少ない場合）
        if  index_finger.y < thumb.y and \
            index_finger.y < middle_finger.y and  \
            index_finger.y < ring_finger.y and  \
            index_finger.y < pinky_finger.y:
            return 'drow'
        else:
            return 'none'
    else:
        return 'none'

def get_status_2(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark
        thumb = hand_landmarks[3]           # 親指
        index_finger = hand_landmarks[8]    # 人差し指
        middle_finger = hand_landmarks[12]  # 中指
        ring_finger = hand_landmarks[16]    # 薬指
        pinky_finger = hand_landmarks[20]   # 小指
        # 親指が他の指より上にある場合（上部からのパーセンテージが少ない場合）
        if  thumb.y < index_finger.y and \
            thumb.y < middle_finger.y and  \
            thumb.y < ring_finger.y and  \
            thumb.y < hand_landmarks[6].y and  \
            thumb.y < hand_landmarks[10].y and  \
            thumb.y < hand_landmarks[14].y and  \
            thumb.y < hand_landmarks[18].y and  \
            thumb.y < pinky_finger.y:
            return 'delete'
        else:
            return 'none'
    else:
        return 'none'

# landmarkの繋がり表示用
landmark_line_ids = [ 
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # 掌
    (1, 2), (2, 3), (3, 4),         # 親指
    (5, 6), (6, 7), (7, 8),         # 人差し指
    (9, 10), (10, 11), (11, 12),    # 中指
    (13, 14), (14, 15), (15, 16),   # 薬指
    (17, 18), (18, 19), (19, 20),   # 小指
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,                # 最大検出数
    min_detection_confidence=0.7,   # 検出信頼度
    min_tracking_confidence=0.7     # 追跡信頼度
)

last_triggerd_at=0
list_status=[[0,0]]
cap = cv2.VideoCapture(0)   # カメラのID指定
if cap.isOpened():
    while True:
        
        # カメラから画像取得
        success, img = cap.read()
        if not success:
            continue
        img = cv2.flip(img, 1)          # 画像を左右反転
        img_h, img_w, _ = img.shape     # サイズ取得

        # 検出処理の実行
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            # 検出した手の数分繰り返し
            for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):

                # landmarkの繋がりをlineで表示
                for line_id in landmark_line_ids:
                    # 1点目座標取得
                    lm = hand_landmarks.landmark[line_id[0]]
                    lm_pos1 = (int(lm.x * img_w), int(lm.y * img_h))
                    # 2点目座標取得
                    lm = hand_landmarks.landmark[line_id[1]]
                    lm_pos2 = (int(lm.x * img_w), int(lm.y * img_h))
                    # line描画
                    cv2.line(img, lm_pos1, lm_pos2, (128, 0, 0), 1)

                # landmarkをcircleで表示
                z_list = [lm.z for lm in hand_landmarks.landmark]
                z_min = min(z_list)
                z_max = max(z_list)
                for lm in hand_landmarks.landmark:
                    lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
                    lm_z = int((lm.z - z_min) / (z_max - z_min) * 255)
                    cv2.circle(img, lm_pos, 10, (255, lm_z, lm_z), -1)

                # 検出情報をテキスト出力
                # - テキスト情報を作成
                hand_texts = []
                for c_id, hand_class in enumerate(results.multi_handedness[h_id].classification):
                    hand_texts.append("#%d-%d" % (h_id, c_id)) 
                    hand_texts.append("- Index:%d" % (hand_class.index))
                    hand_texts.append("- Label:%s" % (hand_class.label))
                    hand_texts.append("- Score:%3.2f" % (hand_class.score * 100))
                # - テキスト表示に必要な座標など準備
                lm = hand_landmarks.landmark[0]
                lm_x = int(lm.x * img_w) - 50
                lm_y = int(lm.y * img_h) - 10
                lm_c = (64, 0, 0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # - テキスト出力
                for cnt, text in enumerate(hand_texts):
                    cv2.putText(img, text, (lm_x, lm_y + 10 * cnt), font, 0.3, lm_c, 1)

                # finger_pose=detectFingerPose(hand_landmarks)
                # print(finger_pose)

                # now=time.time()
                # delta_time=now - last_triggerd_at
                status=get_status(results)
                status_2=get_status_2(results)
                

                if status in ["drow"] :
                    im=hand_landmarks.landmark[8]
                    im_pos = (int(im.x * img_w), int(im.y * img_h))
                    list_status.append(im_pos)
                  
                elif status_2 in ["delete"] :
                    list_status.clear()
                    list_status=[[0,0]]

                print(status)


                len_status=len(list_status) 
                len_status_1=len_status-1     
                # cv2.circle(img, list_status[len_status-1], 20,(0,0,255),3)            
                status_center=list_status[len_status -1]
                # print(status_center)
                if status_center != 0:
                    for ls in list_status:
                        cv2.circle(img, ls, 10,(0,0,255),-1)
                    
    
        # 画像の表示
        cv2.imshow("MediaPipe Hands", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 0x1b:
            break


cap.release()