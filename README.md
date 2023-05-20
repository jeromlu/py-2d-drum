# Simple script to calculate egien values of 2D drum


## Prepare environment

1) Install:
    - [Git](https://git-scm.com/downloads) - to get the code
    - [Python 3.9 or latter.](https://www.python.org/) (make sure it in the Environmental path)
    - [TexLive](https://www.tug.org/texlive/windows.html) - for the report
    - Use `PowerShell` in next steps.

2) Clone repository - [link](https://github.com/jeromlu/py-2d-drum):
```powershell
git clone 'git@github.com:jeromlu/py-2d-drum.git'
```

3) Move to the cloned folder `py-2d-drum`


4) Create python environment (creates folder `venv` in project folder):
```powershell
python -m venv venv
```

5) Activate the environment:
```powershell
./venv/Scripts/activate
```

6) Install required python packages (be sure you did activate the environment):
```powershell
pip install numpy matplotlib scipy
```
