import sys


def pprogress(percent, suffix_message=None, size=50):
    """
    Print a progress bar
    percent = -1 to end and remove the progress bar
    """

    # If process is finished
    if percent == -1:
        sys.stdout.write("\r" + " " * 80 + "\r\r")
        sys.stdout.flush()
        return

    # Compute current progress
    progress = (percent + 1) * size / 100

    # Build progress bar
    bar = "["
    for i in range(int(progress) - 1):
        bar += "="
    bar += ">"
    for i in range(size - int(progress)):
        bar += " "
    bar += "]"

    if suffix_message:
        bar += "  " + suffix_message

    # Write progress bar
    sys.stdout.write("\r%d%% " % (percent + 1) + bar)
    sys.stdout.flush()
