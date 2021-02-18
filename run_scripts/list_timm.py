import timm
from pprint import pprint
model_names = timm.list_models(pretrained=True)
pprint(model_names)