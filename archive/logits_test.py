class A:
    @classmethod
    def build(cls):
        return cls()

    def a(self):
        self.b()

    def b(self):
        print("A.b()")


class B(A):

    def b(self):
        print("B.b()")


b = B.build()
b.a()
