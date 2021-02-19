import time

def print_div(text, custom='', end='\n\n'):
    print('-'*50, '\n')
    print(f"{text}{custom}", end=end)

class timer():
    
    def __init__(self, text):
        print_div(text, '...', end=' ')
        self.t_start = time.time()

    def end(self):
        t_cost = time.time() - self.t_start
        print('\033[35m', end='')
        print('Done ({:.5f}s) {}\n'.format(t_cost, '\033[0m'))

class logger():

    def __init__(self, text):
        print_div(text)
        