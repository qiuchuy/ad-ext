import sys


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    use_color = sys.stdout.isatty()

    @staticmethod
    def color_helper(text, color, num=1, end="\n"):
        print(f"{color}{text}{Colors.ENDC}" * num, end=end)

    @staticmethod
    def print_color(text, color, num=1, end="\n"):
        if Colors.use_color:
            Colors.color_helper(text, color, num, end)
        else:
            print(f"{text}" * num, end=end)


def delimiter(char, color):
    Colors.print_color(char, color, 100)


def center(text, color):
    length = int((100 - len(text)) / 2)
    Colors.print_color(" ", color, length, end="")
    Colors.print_color(text, color, 1)
