#!/usr/bin/env python3
import argparse, json
from train_color_name_model import predict
p=argparse.ArgumentParser()
p.add_argument('hex', help='Color such as #7F4A36')
p.add_argument('--model', default='color_name_model_extended.joblib')
p.add_argument('--top', type=int, default=3)
a=p.parse_args()
print(json.dumps(predict(a.hex, a.model, a.top), indent=2, ensure_ascii=False))
