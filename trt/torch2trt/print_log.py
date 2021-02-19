import time

class log():

    def __init__(self, text):
        print('-'*50, '\n')
        print(text, end=' ')
        self.t_start = time.time() 

    def end(self):
        end_time = time.time() - self.t_start
        print('Done ({:3f}s)'.format(end_time), '\n')