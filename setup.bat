@echo off
setlocal

:: Prompt for virtual environment name with default 'venv'
set VENV_NAME=ultralytics/ultralytics-venv
set /p VENV_NAME="Enter the name for your virtual environment (Press Enter for default 'ultralytics/ultralytics-venv'): "

:: Create the virtual environment
echo Creating virtual environment named %VENV_NAME%...
python3.11 -m venv %VENV_NAME%

:: Add .gitignore to the virtual environment folder
echo Creating .gitignore in the %VENV_NAME% folder...
(
echo # Ignore all content in the virtual environment directory
echo *
echo # Except this file
echo !.gitignore
) > "%VENV_NAME%\.gitignore"

:: Create directories
echo Creating project directories...
for %%D in (generate_input generate_output models training_output dataset dataset\test dataset\train dataset\valid) do (
    if not exist %%D (
        mkdir %%D
    )
)

:: Add .gitignore to the created directories
echo Creating .gitignore files in the project directories...
for %%D in (generate_input generate_output models training_output dataset dataset\test dataset\train dataset\valid) do (
    (
    echo # Ignore all files in this directory
    echo *
    echo # Except this file
    echo !.gitignore
    ) > %%D\.gitignore
)

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
