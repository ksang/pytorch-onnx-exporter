import argparse
import os
from operator import attrgetter
from urllib.parse import urlparse
from pathlib import Path

import torch.utils.model_zoo as model_zoo
import torch.onnx
import torchvision

def is_url(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except:
        return False

def open_model(model_path, args):
    if args.hub_repo != "":
        model = torch.hub.load(args.hub_repo, model_path, pretrained=True)
        output_name = "{}_pytorch_hub".format(model_path)
    elif is_url(model_path):
        model = model_zoo.load_url(model_path)
        output_name = "downloaded"
    elif os.path.isfile(model_path):
        model = torch.load(model_path)
        print(model.keys())
        output_name = "{}_local".format(Path(model_path).resolve().stem)
    else:
        model = attrgetter(model_path)(torchvision.models)(pretrained=True)
        output_name = "{}_torchvision".format(model_path)
    return model, output_name

def export_model(model, args):
    export_params = not args.without_parameters
    output = os.path.join("output", args.output)
    if not os.path.exists('output'):
        os.makedirs('output')
    
    inputs = torch.randn(args.batch_size, args.channel, args.height, args.width)
    print("Model:")
    print(model)
    print("Exporting onnx graph to: {}".format(output))
    # Export the model
    torch.onnx.export(model,                    # model being run
                    inputs,                          # model input (or a tuple for multiple inputs)
                    output,                     # where to save the model (can be a file or file-like object)
                    export_params = export_params,        # store the trained parameter weights inside the model file
                    opset_version = 11,           # the ONNX version to export the model to
                    do_constant_folding = True,   # whether to execute constant folding for optimization
                    input_names = ['input'],    # the model's input names
                    output_names = ['output'],  # the model's output names
                    dynamic_axes = {'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export pytorch model as onnx graph')
    parser.add_argument('-m', '--model', type=str, required=True, help='The model name to export, can be local file, url or torchvision model name')
    parser.add_argument('-n', '--batch-size', type=int, default=1, help='The batch size used for inputs')
    parser.add_argument('--width', type=int, default=640, help='The image width of model inputs')
    parser.add_argument('--height', type=int, default=480, help='The image neight of model inputs')
    parser.add_argument('--channel', type=int, default=3, help='The image channel number of model inputs')
    parser.add_argument('-o', '--output', type=str, default="", help='Overwrite output filename, default is \"output/<model_name>_<source>_<input_shape>.onnx\"')
    parser.add_argument('-p', '--without-parameters', action='store_true', help='Do not store parameters along with the graph')
    parser.add_argument('-t', '--training-mode', action='store_true', help='Export training mode graph rather than inference model')
    parser.add_argument('--hub-repo', type=str, default="", help='PyTorch Hub repo dir for the model')

    args = parser.parse_args()

    model, output_name = open_model(args.model, args)
    if not args.training_mode:
        model.eval()

    if args.output == "":
        args.output = "{}_{}x{}x{}.onnx".format(output_name, args.channel, args.height, args.width)

    export_model(model, args)