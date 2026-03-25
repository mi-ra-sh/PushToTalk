Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\PushToTalk"
WshShell.Run """C:\Users\mihai\AppData\Local\Programs\Python\Python311\python.exe"" ""C:\PushToTalk\push_to_talk.py""", 0, False
