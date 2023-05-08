# neural-sentry

## Getting Started
1. Install dependencies
  - `pip install -r requirements.txt`
2. Configure your GitHub access token
3. Set up Developer Environment
  - Install Python extension in VSCode
  - Configure GitHub access token
    1. Create a [GitHub access token](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token)
    1. Copy the `.env.example` file to `.env`
    1. Replace the `GITHUB_ACCESS_TOKEN` value with your access token
  - (Optional) Create a virtual environment with [pyenv](https://realpython.com/intro-to-pyenv/)
  - Run Python Notebook
    1. Using VSCode
    1. Using [Jupyter](https://jupyter.org/install)



### Create a virtual environment
```bash
python -m venv venv

# linux
source venv/bin/activate

# windows
venv\Scripts\activate

# vscode
# - if you are using vscode, you can select the virtual environment by clicking on the python version in the bottom left corner
# - or press `ctrl+shift+p` and type `Python: Select Interpreter`
# - this requires the python extension to be installed
```
