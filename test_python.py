class robot:
  name = 'robot'
  age = 0
  def __init__(self, name, age):
    print('robot 생성자 호출!')
    self.name = name
    self.age = age
  def __del__(self):
    print('robot 소멸자 호출!')
  def info(self):
    print('나의 이름은', self.name, '입니다!')
    print('나이는', self.age, '입니다!')

class strong_robot(robot):
  weapon = 'gun'
  def __init__(self, name, age, weapon):
    print('strong_robot 생성자 호출!')
    super().__init__(name, age)
    #self.__init__(name, age)
    self.weapon = weapon
  def info(self): #오버라이딩
    super().info()
    print(self.weapon, '로 싸웁니다!')
  #def __del__(self):
  #    print('strong_robot 소멸 자호출!')

s = strong_robot('hulk', 3, 'sword')
#s.info()