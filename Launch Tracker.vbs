Option Explicit
' Launch Tracker.vbs
' Double-click this any time to open the C. elegans Tracker GUI.
' No typing, no visible command window -- just opens the app.
' Run Setup.vbs first if you haven't already (one time only).

Dim shell, fso, scriptDir, exitCode

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = scriptDir

' Quick check that Python is available before trying to launch
exitCode = shell.Run("cmd /c python --version >nul 2>&1", 0, True)
If exitCode <> 0 Then
    MsgBox "Python wasn't found on this computer." & vbCrLf & vbCrLf & _
           "Please double-click 'Setup' first (or install Python from " & _
           "https://www.python.org/downloads/ if you haven't already).", _
           vbExclamation, "C. elegans Tracker"
    WScript.Quit
End If

' Run the GUI with plain `python` (the same interpreter run_gui.bat uses) in a
' hidden window. We used to call `pythonw` here, which on some installs
' (Microsoft Store Python, multiple Pythons) isn't the same Python as `python`
' and fails with no message at all. Waiting for it to exit lets us tell you if
' it failed to start, instead of silently doing nothing.
exitCode = shell.Run("cmd /c python tracker_gui.py", 0, True)
If exitCode <> 0 Then
    MsgBox "The tracker GUI closed with an error (code " & exitCode & ")." & vbCrLf & vbCrLf & _
           "Double-click run_gui.bat in this folder to see the full error message.", _
           vbExclamation, "C. elegans Tracker"
End If
