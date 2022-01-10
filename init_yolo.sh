:'
# Create libs directory
echo "Create libs directory"
rm -rf libs
mkdir libs
cd libs
# Clone keras yolo
git clone https://github.com/qqwweee/keras-yolo3.git
cd keras-yolo
'

# Download yolo weights
cd libs/keras-yolo3
echo "Download yolo weights"
wget -nc https://pjreddie.com/media/files/yolov3.weights
echo "Convert model to keras model"
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
