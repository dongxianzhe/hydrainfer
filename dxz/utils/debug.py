def print_once(message: str):
    if not hasattr(print_once, 'has_printed'):
        print_once.has_printed = False  # 初始化属性
    if not print_once.has_printed:
        print(message)
        print_once.has_printed = True  # 更新状态