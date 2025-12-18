# Installation

This is a quick guide to (re)installing Qudi and these modules for an experiment. We do it in a slightly diffrent way from the original [Ulm IQO's recommandation](iqo-docs/installation_guide). This is mostly a copy of the relevant section of the [Crash course tutorial](Crash course into Qudi).


## Pre-requisites

- We use the `uv` package manager. Please read and follow the [getting started](https://docs.astral.sh/uv/getting-started/) guide. **Windows users: see below how to install it.**
- We will need `git`. This is a version control system (a tool to work collaboratively on code. For example, this is the tool behind [GitHub](https://github.com)). Please [install it](https://git-scm.com/install/). If you want to learn more, you might read [this short guide](https://learnxinyminutes.com/git/) or look for a more extensive tutorial.
- A text editor (optional). In theory the simplest notepad is enough, but you may find more advanced applications useful. If you just need somethin lightweight, you can use [Notepad++](https://notepad-plus-plus.org/). 

### Windows users: installing uv

To install `uv`, you need to use the native Windows shell. Press the {kbd}`Windows` key to open the start menu, then type cmd to bring the native Windows shell. Then, you need to copy the following code:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
and paste it in the Windows shell using right click. Finally press {kbd}`Enter` to execute it. It will download the `uv` executable and make it accessible in `C:/Users/<you>/.local/bin`. To check that everyting worked, open Git Bash and type `uv help` followed by {kbd}`Enter`. You should see the following:
```
An extremely fast Python package manager.

Usage: uv [OPTIONS] <COMMAND>

Commands:
  auth                       Manage authentication
  run                        Run a command or script
  init                       Create a new project
  add                        Add dependencies to the project
  remove                     Remove dependencies from the project
  version                    Read or update the project's version
  sync                       Update the project's environment
  lock                       Update the project's lockfile
  export                     Export the project's lockfile to an alternate format
  tree                       Display the project's dependency tree
  format                     Format Python code in the project
  tool                       Run and install commands provided by Python packages
  python                     Manage Python versions and installations
  pip                        Manage Python packages with a pip-compatible interface
  venv                       Create a virtual environment
  build                      Build Python packages into source distributions and wheels
  publish                    Upload distributions to an index
  cache                      Manage uv's cache
  self                       Manage the uv executable
  generate-shell-completion  Generate shell completion
  help                       Display documentation for a command

Cache options:
  -n, --no-cache               Avoid reading from or writing to the cache, instead using a temporary directory for the duration of the operation [env: UV_NO_CACHE=]
      --cache-dir <CACHE_DIR>  Path to the cache directory [env: UV_CACHE_DIR=]

Python options:
      --managed-python       Require use of uv-managed Python versions [env: UV_MANAGED_PYTHON=]
      --no-managed-python    Disable use of uv-managed Python versions [env: UV_NO_MANAGED_PYTHON=]
      --no-python-downloads  Disable automatic downloads of Python. [env: "UV_PYTHON_DOWNLOADS=never"]

Global options:
  -q, --quiet...                                   Use quiet output
  -v, --verbose...                                 Use verbose output
      --color <COLOR_CHOICE>                       Control the use of color in output [possible values: auto, always, never]
      --native-tls                                 Whether to load TLS certificates from the platform's native certificate store [env: UV_NATIVE_TLS=]
      --offline                                    Disable network access [env: UV_OFFLINE=]
      --allow-insecure-host <ALLOW_INSECURE_HOST>  Allow insecure connections to a host [env: UV_INSECURE_HOST=]
      --no-progress                                Hide all progress outputs [env: UV_NO_PROGRESS=]
      --directory <DIRECTORY>                      Change to the given directory prior to running the command [env: UV_WORKING_DIR=]
      --project <PROJECT>                          Discover a project in the given directory [env: UV_PROJECT=]
      --config-file <CONFIG_FILE>                  The path to a `uv.toml` file to use for configuration [env: UV_CONFIG_FILE=]
      --no-config                                  Avoid discovering configuration files (`pyproject.toml`, `uv.toml`) [env: UV_NO_CONFIG=]
  -h, --help                                       Display the concise help for this command
  -V, --version                                    Display the uv version

Use `uv help <command>` for more information on a specific command.
```


### Windows: creating desktop shortcuts

It is quite convenient to simply click to run Qudi. you can do that by manually creating desktop shortcuts.

Right-click on the desktop and click on {kbd}`New > Shortcut`. In the `location` prompt, you can actually paste a command to be run (instead of the path to a directory). That's what we want to do.

```
%windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "cd $HOME\Documents\qudi-lab-on-a-molecule-modules; uv run --no-dev qudi --debug"
```

This is a command to launch a terminal, navigate to the folder wher Qudi is, and run Qudi. It will work provided that you installed Qudi in you `Documents` folder and that the folder is called `qudi-lab-on-a-molecule-modules`. You may also set a nice icon for the shortcut if you right-click on the created shortcut, then go to {kbd}`properties` and click {kbd}`Change icon...`. Then cick on {kbd}`Browse...` and navigate to `Documents\qudi-lab-on-a-molecule-modules\.venv\Lib\site-packages\qudi\artwork\logo`.

To have a shortcut for the Jupyter notebook installation that comes with Qudi, you can follow the same procedure, but instead of having a command that launches Qudi, make it launch the jupyter:

```
%windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "cd $HOME\Documents\qudi-lab-on-a-molecule-modules; uv run jupyter notebook"
```

A nice icon for the shortcut is available in `Documents\qudi-lab-on-a-molecule-modules\.venv\Lib\site-packages\jupyter_server\static\favicon.ico`.

