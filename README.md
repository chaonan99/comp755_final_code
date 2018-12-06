# DL Research Code
The code is organized like this:
* `afflm` is the package name;
* Under package name, there are several experiments. `tflm_bilm_char` is the experiment name;
* Under each experiment, there are 5 basic files (when saying "run this file", it means something like `python xxx`):
    - `config.py` configuration file. **Run this file before you do anything else under this experiment!**;
    - `data.py` dataset definition. Run this file to get an example test of the data;
    - `model.py` model definition. Run this file to test if the model could run correctly and print desired loss;
    - `train.py` training schema. This file tends to be long, but you do not need to make many changes when transferring to other experiments. That is to say, put changing parts into model or data, not train. This should only handle the epoch, log (tensorboard summary if you use Tensorflow), checkpoint stuff and so on;
    - `predict.py` run test.
* Each experiment may have several runs, specified by the run name. The training script will create a folder for a single run name under `dump` folder, to save the options, log, checkpoints and other things;
* All other files are treated as utilities and should be put into `package_name/utils` folder.
