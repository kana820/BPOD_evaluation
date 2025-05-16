from colorama import init, Style, Fore

# Initialize Colorama.
init()


def cyan(x: str) -> str:
    return f"{Fore.CYAN}{x}{Style.RESET_ALL}"


def green(x: str) -> str:
    return f"{Fore.GREEN}{x}{Style.RESET_ALL}"


def yellow(x: str) -> str:
    return f"{Fore.YELLOW}{x}{Style.RESET_ALL}"


def red(x: str) -> str:
    return f"{Fore.RED}{x}{Style.RESET_ALL}"


def gray(x: str) -> str:
    return f"{Fore.LIGHTBLACK_EX}{x}{Style.RESET_ALL}"