Set WshShell = CreateObject("WScript.Shell")
pyPath = WshShell.ExpandEnvironmentStrings("%LOCALAPPDATA%\Programs\Python\Python311\pythonw.exe")
WshShell.CurrentDirectory = "C:\PushToTalk"
WshShell.Run """" & pyPath & """ ""C:\PushToTalk\push_to_talk.py""", 0, False
