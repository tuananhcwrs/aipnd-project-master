# Predict flower name from an image with predict.py along with the probability of that name.
# That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

# Basic usage: python predict.py /path/to/image checkpoint
# Options:
# * Return top # K most likely classes: python predict.py input checkpoint --top_k 3 
# * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json 
# * Use GPU for inference: python predict.py input checkpoint --gpu

from utils import load_checkpoint, set_device, predict

import argparse
arg_parser = argparse.ArgumentParser('predict.py')


arg_parser.add_argument('input', nargs='?', action="store", default = 'flowers/test/1/image_06752.jpg', type = str)
arg_parser.add_argument('checkpoint', nargs='?', action="store", default = '.', type = str)
arg_parser.add_argument('--top_k', dest = "top_k", action = "store", default = 5, type = int)
arg_parser.add_argument('--category_names', dest = "category_names", action = "store", default = 'cat_to_name.json', type = str)
arg_parser.add_argument('--gpu', dest = "gpu", action = "store", default = "gpu", type = str)

params = arg_parser.parse_args()

### Label mapping
import json

with open(params.category_names, 'r') as f:
    cat_to_name = json.load(f)

print("\nLabel mapping is now complete!")

device = set_device(params.gpu)

model, _, _ = load_checkpoint(device, params.checkpoint)

probs, classes = predict(params.input, cat_to_name, model, device, params.top_k)
    
# Print prediction results
class_names = [cat_to_name[i] for i in classes]
for i in range(len(probs)):
    print(f"{class_names[i]} ({round(probs[i], 3)})")

print("\nFinish prediction!")
