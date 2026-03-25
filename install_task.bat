@echo off
echo Creating PushToTalk scheduled task...
schtasks /create /tn "PushToTalk" /tr "wscript.exe \"C:\PushToTalk\push_to_talk_silent.vbs\"" /sc onlogon /rl highest /f
if %errorlevel% equ 0 (
    echo.
    echo SUCCESS! Task created.
    echo PushToTalk will start automatically at login.
    echo.
    echo To test now, run:
    echo   schtasks /run /tn "PushToTalk"
) else (
    echo.
    echo ERROR: Could not create task. Make sure you run this as Administrator.
)
pause
