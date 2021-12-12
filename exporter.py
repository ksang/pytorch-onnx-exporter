import argparse
import os
from operator import attrgetter
from urllib.parse import urlparse

import torch.utils.model_zoo as model_zoo
import torch.onnx
import torchvision

def is_url(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False

def open_model(model_path):
    if is_url(model_path):
        model = model_zoo.load_url(model_path)
        output_name = "downloaded"
    elif os.path.isfile(model_path):
        model = torch.load(model_path)
        output_name = "local"
    else:
        model = attrgetter(model_path)(torchvision.models)(pretrained=True)
        output_name = "{}_torchvision".format(model_path)
    return model, output_name

def export_model(model, batch_size, output, without_parameters):
    x = torch.randn(batch_size, 3, 224, 224)
    export_params = not without_parameters
    output = os.path.join("output", output)
    print("Exporting onnx graph to: {}".format(output))
    # Export the model
    torch.onnx.export(model,                    # model being run
                    x,                          # model input (or a tuple for multiple inputs)
                    output,                     # where to save the model (can be a file or file-like object)
                    export_params=export_params,        # store the trained parameter weights inside the model file
                    opset_version=10,           # the ONNX version to export the model to
                    do_constant_folding=True,   # whether to execute constant folding for optimization
                    input_names = ['input'],    # the model's input names
                    output_names = ['output'],  # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export pytorch model as onnx graph')
    parser.add_argument('-m', '--model', type=str, required=True, help='The model name to export, can be local file, url or torchvision model name')
    parser.add_argument('-n', '--batch-size', type=int, default=1, help='The batch size used when exporting model')
    parser.add_argument('-o', '--output', type=str, default="", help='Output filename, will be \"<source>_<specs>.onnx\" if not provided')
    parser.add_argument('-p', '--without-parameters', action='store_true', help='Do not store parameters along with the graph')
    parser.add_argument('-t', '--training-mode', action='store_true', help='Export training mode graph rather than inference model')
    args = parser.parse_args()

    model, output_name = open_model(args.model)
    if not args.training_mode:
        model.eval()

    if args.output == "":
        args.output = "{}.onnx".format(output_name)
    
    export_model(model, args.batch_size, args.output, args.without_parameters)