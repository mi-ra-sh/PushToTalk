@echo off
chcp 65001 >nul
cd /d C:\PushToTalk
set PYTHONIOENCODING=utf-8
"C:\Users\mihai\AppData\Local\Programs\Python\Python311\python.exe" -u push_to_talk.py >> C:\PushToTalk\debug.log 2>&1
