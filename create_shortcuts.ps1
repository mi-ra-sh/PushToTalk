$ws = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')

$presets = @(
    @{Name='VM Whisper'; Bat='C:\PushToTalk\preset_whisper.bat'; Key='Ctrl+Alt+1'},
    @{Name='VM Gaming';  Bat='C:\PushToTalk\preset_gaming.bat';  Key='Ctrl+Alt+2'},
    @{Name='VM Music';   Bat='C:\PushToTalk\preset_music.bat';   Key='Ctrl+Alt+3'}
)

foreach ($p in $presets) {
    $path = Join-Path $desktop ($p.Name + '.lnk')
    $sc = $ws.CreateShortcut($path)
    $sc.TargetPath = $p.Bat
    $sc.WorkingDirectory = 'C:\PushToTalk'
    $sc.WindowStyle = 7
    $sc.Hotkey = $p.Key
    $sc.Save()
    Write-Host "Created: $path ($($p.Key))"
}
