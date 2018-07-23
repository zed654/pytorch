class a:
  a_value = 33
  def __init__(self):
    print("a 생성자")
  def gnp(self):
    print("handsome")


class b(a):
  def __init__(self):
    print("b 생성자")

  def gnp(self):
    print(self.a_value)
    print("존나")
    super().gnp()
    return 3


class c(a):
  def __init__(self):
    print("c 생성자")

b_m = b()

# print 1
b.gnp(b_m)

# print 2
print(b.a_value)

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