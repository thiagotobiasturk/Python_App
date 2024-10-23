import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.clock import Clock
import cv2
import numpy as np
import tensorflow as tf

class ObjectRecognitionApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the TFLite model
        self.model = tf.lite.Interpreter(model_path="model/detect_2.tflite")
        self.model.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.labels = self.load_label_map("model/labelmap.txt")  # Load labels

    def load_label_map(self, label_map_path):
        """Load the label map from a text file."""
        with open(label_map_path, 'r') as f:
            labels = [line.strip() for line in f.readlines() if line.strip() and line.strip() != '???']
        print("Number of labels loaded:", len(labels))  # Debug print
        return labels

    def build(self):
        self.img_widget = Image()
        layout = BoxLayout()
        layout.add_widget(self.img_widget)
        # Start the camera capture loop
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)
        self.capture = cv2.VideoCapture(0)  # Change 0 to 1, 2, etc. to use a different camera
        return layout

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Object recognition logic using TFLite model
            detections = self.detect_objects(frame)

            # Draw boxes around detected objects
            for box, class_id in detections:
                ymin, xmin, ymax, xmax = box
                (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                              ymin * frame.shape[0], ymax * frame.shape[0])
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                
                # Display the class label
                if int(class_id) < len(self.labels):
                    label = self.labels[int(class_id)]
                else:
                    label = "Unknown"  # Handle out-of-range class ID
                cv2.putText(frame, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert to Kivy texture
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img_widget.texture = texture

    def detect_objects(self, frame):
        # Prepare the image for the TFLite model
        input_shape = self.input_details[0]['shape']
        input_tensor = cv2.resize(frame, (input_shape[2], input_shape[1]))

        input_tensor = np.clip(input_tensor, 0, 255).astype(np.uint8)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

        # Run the model
        self.model.set_tensor(self.input_details[0]['index'], input_tensor)
        self.model.invoke()

        # Get the output
        boxes = self.model.get_tensor(self.output_details[0]['index'])  # Bounding boxes
        classes = self.model.get_tensor(self.output_details[1]['index'])  # Class IDs
        scores = self.model.get_tensor(self.output_details[2]['index'])  # Scores

        # Print the detected class IDs for debugging
        print("Detected classes:", classes)

        # Filter out detections based on a confidence threshold
        detections = []
        threshold = 0.5  # Confidence threshold
        for i in range(len(scores[0])):
            if scores[0][i] >= threshold:
                detections.append((boxes[0][i], classes[0][i]))

        return detections  # Return the bounding boxes and class IDs

if __name__ == '__main__':
    ObjectRecognitionApp().run()
