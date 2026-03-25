@echo off
echo Removing PushToTalk scheduled task...
schtasks /delete /tn "PushToTalk" /f
if %errorlevel% equ 0 (
    echo.
    echo SUCCESS! Task removed.
    echo PushToTalk will now only start via Startup folder.
) else (
    echo.
    echo ERROR: Could not remove task. Make sure you run this as Administrator.
)
pause
