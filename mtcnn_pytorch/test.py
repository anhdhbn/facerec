from src import detect_faces
from PIL import Image

image = Image.open('images/office1.jpg')
bounding_boxes, landmarks = detect_faces(image)

print("bounding_boxes:\n", bounding_boxes)
print("landmarks:\n", landmarks)
