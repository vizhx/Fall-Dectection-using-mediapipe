import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import socket






def central_difference_rate(y):
    x=range(1,16)
    n = len(x)
    table = [[0] * 6 for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]

    for j in range(1, 6):
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

host = '0.0.0.0' 
port = 12345 

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.setblocking(False)
server_socket.bind((host, port))

print(f"Server listening on {host}:{port}")



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

cap=cv2.VideoCapture(1)
if cap.isOpened():
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps)
    print(width,height)

frame_no=0
angle_arr1=[]
angle_arr2=[]
with mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.5) as pose:
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
                left_hip=np.array([landmarks[23].x,landmarks[23].y])
                right_hip=np.array([landmarks[24].x,landmarks[24].y])
                left_knee=np.array([landmarks[25].x,landmarks[25].y])
                right_knee=np.array([landmarks[26].x,landmarks[26].y])
                left_ankle=np.array([landmarks[27].x,landmarks[27].y])
                right_ankle=np.array([landmarks[28].x,landmarks[28].y])
                #print(left_shoulder,left_shoulder-[0.100,0])
                angle1=None
                angle2=None
                try:
                    angle1=calculate_angle(left_knee,left_ankle,left_ankle-[0.100,0])
                    angle2=calculate_angle(left_shoulder,left_hip,left_hip-[0.100,0])
                    #print(angle2)
                    angle_arr1.append(angle1)
                    angle_arr2.append(angle2)
                    frame_no+=1
                    try:
                        data,address = server_socket.recvfrom(1024)
                        print("Received from {}: {}".format(address,data.decode('utf-8')))
                    except:
                        print("value not received from sensor")

                    if(frame_no==15):
                        rate1=central_difference_rate(angle_arr1)
                        rate2=central_difference_rate(angle_arr2)
                        frame_no=0
                        angle_arr1=[]
                        angle_arr2=[]
                        print(rate1,rate2)
                        if(abs(rate1)>3 and abs(rate2)>5):
                            print('Fall Dectected')
                            break

                    
                    cv2.putText(image,str(angle2), tuple(np.multiply(right_shoulder,[width,height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
                except Exception as e:
                    print(e)
                try:
                    y_value = float(angle1)
                    x_data.append(len(x_data) + 1)  # Increment X-value
                    y_data.append(y_value)
                    line.set_xdata(x_data)
                    line.set_ydata(y_data)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.flush_events()
                except ValueError:
                    print("angle not valid")
            except:
                pass

            mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            cv2.imshow('Mediapipe feed',image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

server_socket.close()
