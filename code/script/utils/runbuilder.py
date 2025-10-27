from collections import namedtuple
from itertools import product  # 给定多个列表输入，可以计算一个笛卡尔积的函数


# RunBuilder类，构建定义我们运行的参数集
class RunBuilder:
    # 这是一个静态方法get_runs,这个方法将根据我们传递的参数来获得它的RUN
    # 静态方法表名不需要类的实例来调用方法
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs
