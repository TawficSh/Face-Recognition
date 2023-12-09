import tensorflow
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

def send_email_with_image(image_path):
    sender_email = "tawfiq.good1@gmail.com"
    sender_password = "cvrqbtywlkgnzfcb"
    receiver_email = "taw20200499@std.psut.edu.jo"
    subject = "Unknown Face Detected"
    body = "An unknown face was detected using your computer. Please find the attached image for reference."

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    with open(image_path, "rb") as image_file:
        image_part = MIMEImage(image_file.read(), name="unknown_face.jpg")
        message.attach(image_part)

    text = message.as_string()

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    try:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, text)
        print("Email sent successfully!")

    except Exception as e:
        print("An error occurred while sending the email:", str(e))

    finally:
        server.quit()

def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)
    return image

batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\ACER\IdeaProjects\ece5831\Images',
    target_size=(160, 160),
    batch_size=batch_size,
    class_mode='binary'
)

class_indices = train_generator.class_indices
print("Class indices:", class_indices)

def recognize_faces():
    detector = MTCNN()
    face_recognition_model = load_model('path/to/your/model.h5')

    cap = cv2.VideoCapture(0)
    email_sent = False

    while True:
        ret, frame = cap.read()

        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, w, h = face['box']
            roi = frame[y:y+h, x:x+w]

            processed_face = preprocess_image(roi)

            label = face_recognition_model.predict(processed_face)[0][0]
            print("Label:", label)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            if label == 1:
                cv2.putText(frame, "Tawfic", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imwrite("unknown_face.jpg", frame)

                if not email_sent:
                    send_email_with_image("unknown_face.jpg")
                    email_sent = True

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss=tensorflow.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

model.fit(train_generator, epochs=10)

model.save('path/to/your/model.h5')

recognize_faces()
