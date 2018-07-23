class A:
  def __init__(self):
    print("Class A__init__()")


class B(A):
  def __init__(self):
    print("Class B__init__()")
    A.__init__(self)

class C(A):
  def __init__(self):
    print("Class C__init__()")
    A.__init__(self)

class D(B, C):
  def __init__(self):
    print("Class D__init__()")
    B.__init__(self)
    C.__init__(self)

d = D()