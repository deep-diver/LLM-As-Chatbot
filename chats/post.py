def strip_pong(ppmanager):
    ppmanager.pingpongs[-1].pong = ppmanager.pingpongs[-1].pong.strip()
    return ppmanager