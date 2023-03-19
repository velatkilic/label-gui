# Label GUI
Label-GUI is a lightweight graphical user interface based on PyQt5 and pyqtgraph for visualizing image or video data and labelling them.

![](/imgs/gui.png)

- Install dependencies
	```
	pip install -r requirements.txt
	```
- Navigate into the src directory and run from the command line
	```
	python app.py
	```
After the class label is entered into the textbox area, pressing the **Add** button creates a color dialog which determines the bounding box color.
![](/imgs/add_label.png)

Mouse scroll can be used to zoom-in and to label objects using left mouse clicks.
![](/imgs/cluster_label.png)

Contrast enhancement can be useful to better determine object boundaries.
![](/imgs/contrast.png)

### Known issues and todo:
- [ ] Self-supervised object detection
- [ ] Segmentation mask painter
