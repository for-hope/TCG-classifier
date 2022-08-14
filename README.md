# TCG Cards Classifier
![image](https://user-images.githubusercontent.com/23053435/184536825-89cb57f2-ee41-40b2-81df-416cb87aa01c.png)


## Overview
TCG-Classifier is a Python library for classifying Card images of the Trading Card Games YuGiOh, Magic The Gathering and Pokemon TCG.

You can view the full approach on [this paper](https://drive.google.com/file/d/15SQoF9Ar1pnvECk0twOfrr1_9sND3Wak/view?usp=sharing).

![image](https://user-images.githubusercontent.com/23053435/184536708-2fcdd7db-4ec2-4f91-bf97-7c9a2c0b6b16.png)
![image](https://user-images.githubusercontent.com/23053435/184536725-a8796573-a2ff-48e8-9630-aa8c749b4a37.png)

## Installation

**1** - Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.txt (You could also use conda)

```bash
pip install -r requirements.txt
```

**2** - Download the training and testing datasets from [here](https://drive.google.com/file/d/1pNeSXZDUwEY7E8KIYIG03qQsX6vMd62H/view?usp=sharing) (~350mb) then extract it on the root of the project to replace the `./data` directory.

**3** - (__OPTIONAL__) you can download pre-trained model from [here](https://drive.google.com/file/d/1KiZSRAA34yyuvxB0fCIvnHPYMtFU_PV6/view?usp=sharing) then extract it to `./models`

## Usage

To classify images run:

```bash
python3 scripts/classifier.py
```
if you want to load a model test data, edit `scripts/classifier.py` like:

```python
start = "load"
dir = 'path/to/card/images'
model_path = 'models/YOUR-MODEL-NAME'

if start == "load":
    ...
```

## Contributing
Maybe you'll be able to contribute soon.

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
