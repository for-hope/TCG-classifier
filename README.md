# TCG Cards Classifier

TCG-Classifier is a Python library for classifying Card images of the Trading Card Games YuGiOh, Magic The Gathering and Pokemon TCG.

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