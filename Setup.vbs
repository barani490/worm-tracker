Option Explicit
' Setup.vbs
' Double-click this ONCE to install everything the tracker needs.
' No typing required -- this just runs the installer silently and
' shows you plain pop-up messages for progress/success/failure.

Dim shell, fso, scriptDir, exitCode

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = scriptDir

' --- Step 1: check Python is installed ---
exitCode = shell.Run("cmd /c python --version >nul 2>&1", 0, True)
If exitCode <> 0 Then
    MsgBox "Python wasn't found on this computer." & vbCrLf & vbCrLf & _
           "Please install it from https://www.python.org/downloads/" & vbCrLf & _
           "During install, check the box that says 'Add Python to PATH'." & vbCrLf & vbCrLf & _
           "Once that's done, double-click Setup again.", _
           vbExclamation, "C. elegans Tracker - Setup"
    WScript.Quit
End If

' --- Step 2: install required packages ---
MsgBox "This will install a few required packages (takes a minute or two)." & vbCrLf & _
       "Click OK to begin.", vbInformation, "C. elegans Tracker - Setup"

exitCode = shell.Run("cmd /c python -m pip install --quiet --disable-pip-version-check -r requirements.txt", 0, True)

If exitCode = 0 Then
    MsgBox "Setup complete!" & vbCrLf & vbCrLf & _
           "You can now double-click 'Launch Tracker' any time to open the program.", _
           vbInformation, "C. elegans Tracker - Setup"
Else
    MsgBox "Something went wrong during setup (error code " & exitCode & ")." & vbCrLf & vbCrLf & _
           "If this keeps happening, open Command Prompt in this folder and run:" & vbCrLf & _
           "    python -m pip install -r requirements.txt" & vbCrLf & _
           "to see the full error message.", vbCritical, "C. elegans Tracker - Setup"
End If
