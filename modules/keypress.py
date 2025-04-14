def check_for_keypress() -> bool:
    """Non-blocking keypress check."""
    try:
        import msvcrt  # Windows
        while msvcrt.kbhit():
            key = msvcrt.getch()
            return key == b'\r'
    except ImportError:
        import sys
        import select
        import termios
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            return key == '\n'
    return False
