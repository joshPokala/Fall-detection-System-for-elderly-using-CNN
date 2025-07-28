import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
from dotenv import load_dotenv
import base64
import subprocess


# Load environment variables for paths
load_dotenv()

# SMTP configuration
SMTP_SERVER = os.getenv('SMTP_SERVER')  # Replace with your SMTP server
SMTP_PORT = 587  # Replace with your SMTP port
SMTP_USERNAME = os.getenv('SMTP_USERNAME')  # Replace with your SMTP username
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')  # Replace with your SMTP password
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')  # Replace with recipient's email address


def send_email_alert(frame):
    # convert frame to a PNG image
    image_data = cv2.imencode('.png', frame)[1].tobytes()

    # create email message
    message = MIMEMultipart()
    message['to'] = RECIPIENT_EMAIL
    message['from'] = SMTP_USERNAME
    message['subject'] = "Fall Detected Alert!"

    body = f"""
    <html>
        <body style="text-align: center; font-family: system-ui, Arial, sans-serif;">
            <h1 style="color: #FF3333;">FALL DETECTED!</h1>
            <h2>A fall has been detected and persisted for more than the specified duration. Please check immediately.</h2>
            <img src="cid:image">
        </body>
    </html>
    """
    message.attach(MIMEText(body, 'html'))

    # image as attachment using contentid

    image = MIMEImage(image_data, 'png')
    image.add_header('Content-ID', '<image>')
    message.attach(image)

    # send email using smtp
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.connect(SMTP_SERVER, SMTP_PORT)
            server.starttls()  
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(message)
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Failed to send email alert: {e}")


model_path = os.getenv("model_path") + "/fall_detection_model.h5"
model = tf.keras.models.load_model(model_path)
fall_status = False
g_prediction = 0


def fall_detection():
    print(fall_status, g_prediction)
    return (fall_status, g_prediction)


cap = None


def start_capture():
    global cap
    if 'ANDROID_DATA' in os.environ:
        cap = cv2.VideoCapture('rtsp://localhost:8080/h264.sdp')
    else:
        cap = cv2.VideoCapture('rtsp://192.168.63.5:8080/h264.sdp')


def generate_frames():
    global fall_status, g_prediction
    fall_detected_count = 0  
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        img = cv2.resize(frame, (256, 256))  
        img = img_to_array(img) / 255.0   
        img = np.expand_dims(img, axis=0)  

        prediction = model.predict(img)
        g_prediction = prediction
        print(f"Raw Prediction: {prediction}")
        
        label = "Fall Detected" if prediction > 0.5 else "No Fall"


        if label == "Fall Detected":
            fall_status = True
            fall_detected_count += 1
            if fall_detected_count >= 30:
                send_email_alert(frame)
                fall_detected_count = 0  
        else:
            fall_status = False
            fall_detected_count = 0

        # display result on video feed
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1, color=(100, 100, 255), thickness=2)
        #cv2.imshow("Fall Detection", frame)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # q to quit feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

