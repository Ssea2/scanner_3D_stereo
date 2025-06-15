import gi
gi.require_version('Aravis', '0.8')  # Assurez-vous que la version correspond à celle installée
from gi.repository import Aravis
import cv2 as cv

"""# Mise à jour de la liste des périphériques
Aravis.update_device_list()
connected_num_device = Aravis.get_n_devices()

print(f"Nombre de périphériques connectés : {connected_num_device}")

cam1 = Aravis.Camera.new("10.216.0.201")

cam1.start_acquisition()
        
# Grab a buffer (frame)
print("Waiting for frame...")
buffer = c  # Wait up to 1 second
        
if buffer:
    print("Frame acquired successfully")
            
    # Get image data from buffer
    image_data = buffer.get_data()
            """

def get_image():
    url = "/dev/video2"

    cap = cv.VideoCapture(url)
    c=0
    while True:
        ret, frame = cap.read()
        
        if ret == True:
            cv.imshow("t", frame)
            cv.waitKey(0)
            cv.destroyAllWindows()
            cv.imwrite(f"img_{c}.jpg",frame)
            c+=1
        else:
            print("pas d'img")
            break

get_image()