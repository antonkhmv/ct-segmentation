# ct-segmentation

CT image segmentation using DL in Pytorch

# Train

to train the model, install python3 and the requriements from the file requirements.txt

```bash
python3 -m pip install -r requirements.txt
```

train_model.py - train model

examples/2d/train_config - train model on 2D slices
examples/3d/train_config - train model on 3D volumes

# Inference using nvidia-triton-inference-serving configs:

examples/2d/config.pbtxt - config for 2D model inference
examples/3d/config.pbtxt - config for 3D model inference

# Requests to serving 

client.py

