This is a package I made to allow for a intermediate RFDETR model, one that uses memory efficent xformers package to train and requires significantly less memory. It is called, RFDETRSegIntermediate.

On my local machine, it allowed me to train something that would have taken 16GB into about 8GB of VRAM.

This has 2 fairly different versions of installation. One on Windows and the other on Linux:

### Windows Installation: 
#### Run these BEFORE downloading the package:
```
pip install https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp312-cp312-win_amd64.whl 
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu128
```
Then 
```
pip3 install rfdetr-seg-intermediate
```

### Linux Installation:
```
pip3 install triton==3.0.0
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu128
```

Troubleshooting (how it runs on my local Windows machine):
```
1- run
pip install https://huggingface.co/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp312-cp312-win_amd64.whl 

2- run 
pip install rfdetr-seg-intermediate

3- run
pip install xformers --force-reinstall --index-url https://download.pytorch.org/whl/cu128    
```
### "Why this isnt a 1 click install?"
- Multiple reasons, but there are a few critical packages here that are not even meant to be run on Windows but have been compiled for Win32 thanks to the awesome internet. There are other packages that are in conflict with eachother if installed by 1 click but will be ok if the steps above are followed. 
Honestly, the most critical libraries are the Hugging Face version of Triton and xformers with the proper cuda installed. I try to make my libraries as good as they can, but for this specific one becuase of the abnormalities it makes a 1 click install not possible.

### Other Notes:
- Install all packages EXACTLY as the pyproject.toml wants it to be.
- If for some reason it reinstalls pytorch as the CPU version, install the CUDA version instead if you wish. Just make sure its torch 2.10.0