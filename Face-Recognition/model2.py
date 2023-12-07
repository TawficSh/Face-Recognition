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

# Function to send an email notification with an attached image
def send_email_with_image(image_path):
    # Email details
    sender_email = "tawfiq.good1@gmail.com"
    sender_password = "cvrqbtywlkgnzfcb"
    receiver_email = "taw20200499@std.psut.edu.jo"
    subject = "Unknown Face Detected"
    body = "An unknown face was detected using your computer. Please find the attached image for reference."

    # Create a multipart message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Add body to the email
    message.attach(MIMEText(body, "plain"))

    # Open the image file
    with open(image_path, "rb") as image_file:
        # Attach the image to the email
        image_part = MIMEImage(image_file.read(), name="unknown_face.jpg")
        message.attach(image_part)

    # Convert the message to a string
    text = message.as_string()

    # Connect to the SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    try:
        # Log in to the server
        server.login(sender_email, sender_password)

        # Send the email
        server.sendmail(sender_email, receiver_email, text)
        print("Email sent successfully!")

    except Exception as e:
        print("An error occurred while sending the email:", str(e))

    finally:
        # Disconnect from the server
        server.quit()

# Function to preprocess an image for face recognition
def preprocess_image(image):
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)
    return image

# Set up data generators for training with data augmentation
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

# Define the global train_generator variable
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\ACER\IdeaProjects\ece5831\Images',
    target_size=(160, 160),
    batch_size=batch_size,
    class_mode='binary'
)

class_indices = train_generator.class_indices
print("Class indices:", class_indices)

# Function to recognize faces in real-time using the laptop camera
def recognize_faces():
    # Load the pre-trained MTCNN model for face detection
    detector = MTCNN()

    # Load your trained face recognition model
    face_recognition_model = load_model('path/to/your/model.h5')

    cap = cv2.VideoCapture(0)

    # Flag to track whether an email has been sent
    email_sent = False

    while True:
        ret, frame = cap.read()

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)

        for face in faces:
            x, y, w, h = face['box']
            roi = frame[y:y+h, x:x+w]

            # Preprocess the face for recognition
            processed_face = preprocess_image(roi)

            # Use the face recognition model to predict the label
            label = face_recognition_model.predict(processed_face)[0][0]
            print("Label:", label)

            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Display the name on the frame based on the predicted label
            if label == 1:  # Adjust the threshold as needed
                cv2.putText(frame, "Tawfic", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imwrite("unknown_face.jpg", frame)

                # If an unknown face is detected and email has not been sent, send an email with the captured image
                if not email_sent:
                    send_email_with_image("unknown_face.jpg")
                    email_sent = True

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        # Press 'q' to exit the loop and close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

# Set up your model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification: 1 neuron with sigmoid activation

# Compile the model
model.compile(optimizer='adam', loss=tensorflow.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10)  # Adjust the number of epochs as needed

# Save the trained model
model.save('path/to/your/model.h5')

# Call the function to start face recognition
recognize_faces()