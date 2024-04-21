import ctypes
import os

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if is_admin():
    print("You have admin privileges.")
else:
    print("You do not have admin privileges.")
