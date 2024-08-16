@echo off
setlocal

:: Prompt for Python path with a default value
set PYTHON_PATH=python
set /p PYTHON_PATH="Enter the path to your Python executable (Press Enter for default 'python'): "

:: Prompt for virtual environment name with default 'ultralytics/ultralytics-venv'
set VENV_NAME=ultralytics/ultralytics-venv
set /p VENV_NAME="Enter the name for your virtual environment (Press Enter for default 'ultralytics/ultralytics-venv'): "

:: Create the virtual environment
echo Creating virtual environment named %VENV_NAME%...
"%PYTHON_PATH%" -m venv %VENV_NAME%

:: Create directories
echo Creating project directories...
mkdir generate_input
mkdir generate_output
mkdir models
mkdir training_output
mkdir dataset
mkdir dataset\test
mkdir dataset\train
mkdir dataset\valid

:: Generate the activate_venv.bat file
echo Generating activate_venv.bat...
(
echo @echo off
echo cd %%~dp0
echo set VENV_PATH=%VENV_NAME%
echo.
echo echo Activating virtual environment...
echo call "%%VENV_PATH%%\Scripts\activate"
echo echo Virtual environment activated.
echo cmd /k
) > activate_venv.bat

echo Setup complete. Use 'activate_venv.bat' to activate the virtual environment.

:: Call activate_venv.bat to activate the virtual environment
echo Activating the virtual environment...
call activate_venv.bat

echo Virtual environment %VENV_NAME% is activated. You can reactivate it in the future by running 'activate_venv.bat'.
endlocal
