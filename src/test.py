import tensorflow as tf 
import timm 

model = timm.create_model('eva02_base_patch14_448.mim_in22k_ft_in22k_in1k', pretrained=True, num_classes=0)

print("hej hej")

string = 'virker det?'

with open(f"test.txt", "w") as f:
        f.write(string)