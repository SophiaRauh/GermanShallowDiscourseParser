# GermanShallowDiscourseParser

**This is a slightly modified/updated version of the Shallow Discourse Parser for German by Peter Bourgonje. Please visit https://github.com/PeterBourgonje/GermanShallowDiscourseParser for details.**

## License

This parser is released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
You can find a human-readable summary of the licence agreement here:

https://creativecommons.org/licenses/by-nc-sa/4.0/

If you use this parser for your research, please cite the following:

[Bourgonje, P. (2021). Shallow Discourse Parsing for German. PhD Thesis, Universität Potsdam](https://publishup.uni-potsdam.de/50663).

## Installation & Usage

- Clone this repository 
- Clone the DiMLex repository to your local system (`git clone https://github.com/discourse-lab/dimlex`)
- Clone the PCC repository to your local system (`git clone https://github.com/PeterBourgonje/pcc2.2`)
- Install all required python packages (`pip install -r requirements.txt`)
- Modify the paths in `config.ini` to match your system configuration. The variables you have to modify are `pccdir` and `dimlexdir`. Make sure these point to the locations where you have just downloaded/unzipped/cloned the respective modules.
- Type (`python Parser.py`) in the command line for a general evaluation of the model (starts `evaluate()`)
- If you want to predict connectives in a given text file, type `python Parser.py -c text_file.txt`, then the results are saved in `text_file_results.json`.

