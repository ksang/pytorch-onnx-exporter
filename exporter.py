import argparse
import os
from operator import attrgetter
from urllib.parse import urlparse
from pathlib import Path

import torch.utils.model_zoo as model_zoo
import torch.onnx
import torchvision
from torch.nn import Transformer
from models.unet import UNet

import onnx
from onnxsim import simplify

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
    elif model_path == "transformer":
        model = Transformer(d_model=args.transformer_dim,
                            nhead=args.transformer_heads,
                            num_encoder_layers=args.transformer_encoder_layers,
                            num_decoder_layers=args.transformer_decoder_layers )
        output_name = "transformer_{}dim_{}heads_{}enc_{}dec".format(args.transformer_dim,
                                                                     args.transformer_heads,
                                                                     args.transformer_encoder_layers,
                                                                     args.transformer_decoder_layers   )
    elif model_path == "unet":
        model = UNet(n_channels=3, n_classes=2)
        if args.checkpoint:
            model.load_state_dict(
                torch.load(args.checkpoint,
                        map_location=torch.device('cpu')))
        output_name = model_path
    elif os.path.isfile(model_path):
        model = torch.load(model_path, map_location=torch.device('cpu'))
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
    
    if args.model.find("yolo") >= 0:
        # for yolo, img is square sized and must by multipliers of max stride
        # we are using width here
        inputs = torch.zeros(args.batch_size, args.channel, args.width)
    elif args.model == "transformer":
        inputs = (torch.zeros(args.batch_size, 32, args.transformer_dim), torch.zeros(args.batch_size, 32, args.transformer_dim))
    else:
        inputs = torch.randn(args.batch_size, args.channel, args.height, args.width)
    
    print("Model:")
    print(model)
    print("Exporting onnx graph to: {}".format(output+".onnx"))
    # Export the model
    torch.onnx.export(model,                    # model being run
                    inputs,                          # model input (or a tuple for multiple inputs)
                    output+".onnx",                     # where to save the model (can be a file or file-like object)
                    export_params = export_params,        # store the trained parameter weights inside the model file
                    opset_version = 11,           # the ONNX version to export the model to
                    do_constant_folding = True,   # whether to execute constant folding for optimization
                    input_names = ['input'],    # the model's input names
                    output_names = ['output'],  # the model's output names
                    dynamic_axes = {'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    if args.optimize:
        onnx_model = onnx.load(output+".onnx")
        print("Optimizing model...")
        # convert model
        if args.model == "transformer":
            model_simp, check = simplify(onnx_model, input_shapes={"input": inputs[0].shape})
        else:
            model_simp, check = simplify(onnx_model, input_shapes={"input": inputs.shape})
        assert check, "Simplified ONNX model could not be validated"
        print("Saving optimized model to: {}".format(output+".optimized.onnx"))
        onnx.save(model_simp, output+".optimized.onnx")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export pytorch model as onnx graph')
    parser.add_argument('-m', '--model', type=str, required=True, help='The model name to export, can be local file, url, transformer or torchvision model name')
    parser.add_argument('-n', '--batch-size', type=int, default=1, help='The batch size used for inputs')
    parser.add_argument('--width', type=int, default=1280, help='The image width of model inputs')
    parser.add_argument('--height', type=int, default=720, help='The image neight of model inputs')
    parser.add_argument('--channel', type=int, default=3, help='The image channel number of model inputs')
    parser.add_argument('-o', '--output', type=str, default="", help='Overwrite output filename, default is \"output/<>_<source>_<input_shape>.onnx\"')
    parser.add_argument('-p', '--without-parameters', action='store_true', help='Do not store parameters along with the graph')
    parser.add_argument('-t', '--training-mode', action='store_true', help='Export training mode graph rather than inference model')
    parser.add_argument('--hub-repo', type=str, default="", help='PyTorch Hub repo dir for the model')
    parser.add_argument('--optimize', action='store_true', help='Optmization and simplify model after export')
    parser.add_argument('--checkpoint', type=str, help='Specify checkpoint file for pretained model')
    # Transformer related args
    parser.add_argument('--transformer-dim', type=int, default=512, help='The input dimension for transformer model')
    parser.add_argument('--transformer-heads', type=int, default=8, help='The number of heads for transformer model')
    parser.add_argument('--transformer-encoder-layers', type=int, default=6, help='The number of encoder layers for transformer model')
    parser.add_argument('--transformer-decoder-layers', type=int, default=6, help='The number of decoder layers for transformer model')

    args = parser.parse_args()

    model, output_name = open_model(args.model, args)
    if not args.training_mode:
        model.eval()

    if args.output == "":
        if args.model == "transformer":
            args.output = output_name
        else:
            args.output = "{}_{}x{}x{}".format(output_name, args.channel, args.height, args.width)

    export_model(model, args)