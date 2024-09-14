from argparse import ArgumentParser

import numpy as np
import tqdm
import tritonclient.grpc
import torch

from data import TIFFLoader, NIfTILoader, DICOMLoader
from tqdm.auto import tqdm


model_name = "unet"

input_name = ["INPUT__0"]
output_name = ["OUTPUT__0"]


def get_inputs(inputs):
    inputs = np.expand_dims(inputs, axis=0)
    inputs = inputs.astype(np.float32)
    input0 = tritonclient.grpc.InferInput(input_name[0], inputs.shape, "FP32")
    input0.set_data_from_numpy(inputs)
    return [input0]


def get_outputs():
    output0 = tritonclient.grpc.InferRequestedOutput(output_name[0])
    return [output0]


def make_request(triton_client, image):
    response = triton_client.infer(
        model_name, model_version="1", inputs=get_inputs(image), outputs=get_outputs()
    )
    return torch.from_numpy(response.as_numpy(output_name[0]).copy())


def main():

    parser = ArgumentParser()
    parser.add_argument("-p", "--file-path", type=str, required=True, help="path to image")
    parser.add_argument("-d", "--file-dim", type=str, required=True, help="2d or 3d")
    parser.add_argument("-t", "--file-type", type=str, required=True, help="TIFF, NIfTI or DICOM")
    parser.add_argument("-e", "--endpoint", type=str, required=False, default=None)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    args = parser.parse_args()

    endpoint = args.endpoint or "127.0.0.1:5321"
    triton_client = tritonclient.grpc.InferenceServerClient(url=endpoint)
    config = triton_client.get_model_config(model_name)

    if args.file_type == "TIFF":
        loader = TIFFLoader()
    elif args.file_type == "NIfTI":
        loader = NIfTILoader()
    elif args.file_type == "DICOM":
        loader = DICOMLoader()
    else:
        raise ValueError(f"Unknown type {args.file_type}")

    if args.file_dim == "2d":
        input_image = loader.load_2d_image(args.file_path)
        print(input_image.min(), input_image.max())
    elif args.file_dim == "3d":
        input_image = loader.load_3d_image(args.file_path)
    else:
        raise ValueError(f"Unknown type {args.file_dim}")

    server_dim = "3d" if len(config.config.input[0].dims) == 4 else "2d"

    if server_dim == "3d" and args.file_dim == "3d":
        result = make_request(triton_client, input_image)
        loader.save_3d_image(args.output_path, result)

    elif server_dim == "3d" and args.file_dim == "2d":
        input_image = np.expand_dims(input_image, 0)
        result = make_request(triton_client, input_image)
        loader.save_2d_image(args.output_path, result)

    elif server_dim == "2d" and args.file_dim == "3d":
        result = []
        for split in tqdm(input_image):
            result.append(make_request(triton_client, split))
        result = torch.from_numpy(np.stack(result, axis=0))
        loader.save_3d_image(args.output_path, result)

    elif server_dim == "2d" and args.file_dim == "2d":
        input_image = input_image.unsqueeze(0)
        result = make_request(triton_client, input_image)
        loader.save_2d_image_float(args.output_path, result > 0.5)


if __name__ == "__main__":
    main()
