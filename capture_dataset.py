
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Save photos in the folder used by train.py
SAVE_DIR = BASE_DIR / "DataSet" / "perfume"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not open. Check camera permissions.")
    exit()

count = len(list(SAVE_DIR.glob("*.jpg")))

print("Press SPACE to take a photo.")
print("Press Q to close the camera.")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not capture image.")
        break

    cv2.putText(
        frame,
        f"perfume photos: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("VisionClassify - Perfume Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):
        filename = SAVE_DIR / f"perfume_{count + 1:03d}.jpg"

        if cv2.imwrite(str(filename), frame):
            count += 1
            print("Photo saved:", filename.name)
        else:
            print("Failed to save photo.")

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
print("Total perfume photos:", count)
