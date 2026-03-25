@echo off
echo Creating Push-to-Talk scheduled task with admin rights...
schtasks /create /tn "PushToTalk" /tr "\"%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe\" \"C:\pushtotalk\push_to_talk.py\"" /sc onlogon /rl highest /f
if %errorlevel%==0 (
    echo.
    echo SUCCESS! Task "PushToTalk" created.
    echo It will run automatically at logon with administrator privileges.
) else (
    echo.
    echo ERROR: Failed to create task. Make sure you run this as Administrator.
)
pause
