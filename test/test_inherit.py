# coding: utf-8

"""
测试Pyhton中__init__的继承情况

结论：
    python 中子类覆盖基类的__init__方法，若想使用基类的__init__()方法，需要显示的调用
"""


class FatherClass(object):
    def __init__(self, f):
        self.f = f
        print("call Father class's __init__ method.")


class SubClass(FatherClass):
    def __init__(self, s, f):
        # super.__init__(4)
        # FatherClass.__init__(self, f)
        super(SubClass, self).__init__(f)
        self.s = s
        print("call Sub class's __init__ method.")
    pass


def main():
    fo = FatherClass(1)
    print(fo.f)
    so = SubClass(2, 4)
    # print(so.s)
    print(so.f)


if __name__ == "__main__":
    main()

