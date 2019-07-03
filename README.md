# Reflection Removal from images
Main goal of this team project is separating the reflection and transmission layers from images taken through window or glass. The main algorithm, is to model the ghosted reflection using a double-impulse convolution kernel, and automatically estimate the spatial separation and relative attenuation of the ghosted reflection components and using this to separate reflection and transmission layers.
The project deals with separating reflection and transmission layers from the images taken through glass or window. In this project we model the ghosted reflection using double impulse convolution kernel and automatically estimate the spatial separation and relative attenuation of the ghosted reflection components. The code requires only a single input image. We demonstrate this using real world inputs as well as synthetic inputs.

##Instructions to run the code
This code is tested on the VM provided for assignments

please copy the file GSModel_8x8_200_2M_noDC_zeromean.mat from https://drive.google.com/open?id=17DgQO153MCFL5EXoMUdCZ-EwxnQEPqTw to this folder

usage:
python reflection_removal.py <option> <optional>

option:
    1 -> apples image from paper
    2 -> synthatic image
    3 -> test image( with this provide the image file name as second arguemnt)
    
examples:
python reflection_removal.py 1
python reflection_removal.py 2
python reflection_removal.py 3 test_image.png


