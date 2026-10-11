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

' pythonw = same as python, but with no console window at all
shell.Run "pythonw tracker_gui.py", 0, False
