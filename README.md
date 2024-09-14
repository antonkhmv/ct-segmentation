# CT Segmentation

CT image segmentation using DL in Pytorch. Used in a group project in HSE University

# Train

to train the model, install python3 and the requriements from the file requirements.txt

```bash
python3 -m pip install -r requirements.txt
```

train: 

```bash
python3 train_model.py --config-file <config file> --kaggle-username <username> --kaggle-api-key <api-key>
```

examples/2d/train_config.yaml - train model on 2D slices

examples/3d/train_config.yaml - train model on 3D volumes

# Inference using nvidia-triton-inference-serving configs:

examples/2d/config.pbtxt - config for 2D model inference

examples/3d/config.pbtxt - config for 3D model inference

https://github.com/triton-inference-server/server

# Requests to serving 

client.py

