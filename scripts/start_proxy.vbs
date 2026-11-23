Set objShell = CreateObject("WScript.Shell")
' Start the proxy daemon with auto-restart capability
objShell.Run """E:\Node\node.exe"" ""E:\BST_lab\GIS CLI\scripts\start_proxy_daemon.js""", 0, False
