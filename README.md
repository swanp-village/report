# MRR

## Need to install

[Python 3](https://www.python.org/downloads/)
[Git](https://git-scm.com/)

## OS

recommend for using Linux / macOS

## Download this repository

```bash
git clone https://github.com/nimiusrd/MRR.git
```

## Update this repository

```bash
git fetch origin master
git merge origin master
```

## 仮想環境 / virtual env
ref: https://docs.python.org/3/library/venv.html

### 作成 / make

```bash
python3 -m venv .venv
```

### 有効化 / activate

Run this command before execution.

```bash
. .venv/bin/activate
```

## Install modules

When you use M1 Mac,

```bash
brew install openblas
pip install cython pybind11 pythran numpy
OPENBLAS=$(brew --prefix openblas) CFLAGS="-falign-functions=8 ${CFLAGS}" pip install --no-use-pep517 scipy scikit-learn
```

This program uses numpy, matplotlib, and so on.
Need to install modules from pip.

```bash
pip3 install -r requirements.txt
```

## 実行 / execute

### 最適化 / optimize

configuration file: `src/config/base.py`

```bash
python main.py
```

### シミュレーション / simulate

`-c`オプションは`src/config/simulate`にあるpythonファイルを指定する．

```bash
cd src
python main.py -c two_same_2
```

## Jupyter

### 起動

```bash
jupyter notebook
```
基本的に`notebook/`でファイルを作成する．

## テスト

```bash
cd src
python -m unittest discover -s __tests__
```
