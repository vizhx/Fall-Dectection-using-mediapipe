import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

def central_difference_rate(y):
    x=range(1,31)
    n = len(x)
    table = [[0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]

    k=int(len(x)/2)
    h=x[1]-x[0]
    rate=1/h*((table[k][1]+table[k-1][1])/2-1/12*(table[k-1][3]+table[k-2][3])+1/60*(table[k-2][5]+table[k-3][5]))
    return rate




def calculate_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle =np.abs(radians*180.0/np.pi)

    if angle >180.0:
        angle=360-angle
    return angle

mp_drawing =mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

x_data = []
y_data = []

# Create an initial empty plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
line, = ax.plot(x_data, y_data)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Dynamic Graph')
ax.set_ylim(0, 100)

cap=cv2.VideoCapture("D:\\Capstone Project\\Videos\\fall-03-cam0.mp4")
if cap.isOpened():
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps)
    print(width,height)

frame_no=0
angle_arr=[]
with mp_pose.Pose(min_detection_confidence=0.2,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
            ret,frame =cap.read()

            #change color to brb
            image =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image.flags.writeable=False
            results = pose.process(image)

            image.flags.writeable =True
            image =cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            
            try:
                global landmarks 
                landmarks= results.pose_landmarks.landmark
                left_shoulder=np.array([landmarks[11].x,landmarks[11].y])
                right_shoulder=np.array([landmarks[12].x,landmarks[12].y])
                left_hip=[landmarks[23].x,landmarks[23].y]
                right_hip=[landmarks[24].x,landmarks[24].y]
                left_knee=np.array([landmarks[25].x,landmarks[25].y])
                right_knee=[landmarks[26].x,landmarks[26].y]
                left_ankle=np.array([landmarks[27].x,landmarks[27].y])
                right_ankle=[landmarks[28].x,landmarks[28].y]
                #print(left_shoulder,left_shoulder-[0.100,0])
                angle=None
                try:
                    angle=calculate_angle(left_knee,left_ankle,left_ankle-[0.100,0])
                    angle_arr.append(angle)
                    frame_no+=1
                    if(frame_no==30):
                        print(len(angle_arr))
                        rate=central_difference_rate(angle_arr)
                        frame_no=0
                        angle_arr=[]
                        print(rate)
                        if(abs(rate)>1.5):
                            print('Fall Dectected')
                            break

                    
                    cv2.putText(image,str(angle), tuple(np.multiply(right_shoulder,[width,height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
                except Exception as e:
                    print(e)
                try:
                    y_value = float(angle)
                    x_data.append(len(x_data) + 1)  # Increment X-value
                    y_data.append(y_value)
                    line.set_xdata(x_data)
                    line.set_ydata(y_data)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.flush_events()
                except ValueError:
                    print("Angle not valid")
            except:
                pass

            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Mediapipe feed',image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
