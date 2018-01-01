# -*- coding: utf-8 -*-

import picos as pic
import cvxopt as cvx
import scipy as sp
import scipy.linalg as sl
import numpy as np

def testSolveLP():
    """
    線形計画問題を解く
    CVXOPTの線形計画問題のサンプル:

    min 2.0 * x1 +  1.0 * x2
    s.t. -x1 + x2 <= 1.0
         x1 + x2 >= 2.0
         x2 >= 0.0
         x1 - 2.0 * x2 <= 4.0
    """
    cnp = np.array([[2,  1]])
    c = pic.new_param('c', cnp)

    sdp = pic.Problem()

    x = sdp.add_variable('x', (cnp.shape[1], 1))
    sdp.add_constraint(-x[0] + x[1] <= 1.0)
    sdp.add_constraint(x[0] + x[1] >= 2.0)
    sdp.add_constraint(x[1] >= 0.0)
    sdp.add_constraint(x[0] - 2.0 * x[1] <= 4.0)

    sdp.set_objective('min', c*x)

    print(sdp)
    sdp.solve()
    print(x.value)


def testSolveQP():
    """
    二次計画問題をSDP形式でわざわざ解く
    CVXOPTの二次計画問題のサンプル:

    min 2.0 * x1**2 + x2**2 + x1 * x2 + x1 + x2
    s.t. x1 >= 0.0
         x2 >= 0.0
         x1 + x2 = 1.0
    """

    Anp = np.array([[4, 1],
                    [1, 2]])
    bnp = np.array([[1], [1]])

    Mnp = sl.sqrtm(Anp/2.0) # 行列Anpの平方根

    b = pic.new_param('b', bnp)
    M = pic.new_param('M', Mnp)
    I = pic.new_param('I', sp.eye(bnp.shape[0]) )

    sdp = pic.Problem()
    f0 = sdp.add_variable('f0', (1, 1))
    x = sdp.add_variable('x', (bnp.shape[0], 1))
    sdp.add_constraint((( f0 - b.T*x & (M*x).T ) // (M*x & I))>>0 ) # シュールの補題より
    sdp.add_constraint(x[0] >= 0.0)
    sdp.add_constraint(x[1] >= 0.0)
    sdp.add_constraint(x[0] + x[1] == 1.0)

    sdp.set_objective('min', f0)

    print(sdp)
    sdp.solve()
    print(x.value)

def testSolveMinEigVal():
    """
    最小固有値問題をSDP形式でわざわざ解く:
    A=[4, 1; 1, 2]の最小固有値1.59を求める
    """

    Anp = np.array([[4, 1],
                    [1, 2]])

    A = pic.new_param('A', Anp)
    I = pic.new_param('I', sp.eye(Anp.shape[0]) )

    sdp = pic.Problem()
    mu = sdp.add_variable('mu', (1, 1))
    sdp.add_constraint(A - mu * I >> 0 )
    sdp.set_objective('min', -mu)

    print(sdp)
    sdp.solve()
    print(mu.value)

def testSolveLyapnovEq():
  """
  "LMIによるシステム制御" p.85の例4.3(ロバスト安定性解析)を解いてみる

  P - I >> 0
  P * A1 + A1.T * P << 0
  P * A2 + A2.T * P << 0
  P * A3 + A3.T * P << 0
  """

  A1a = np.array([[-1.0, 2.9], [1.0, -3.0]])
  A2a = np.array([[-0.8, 1.5], [1.3, -2.7]])
  A3a = np.array([[-1.4, 0.9], [0.7, -2.0]])

  A1 = pic.new_param('A1', A1a )
  A2 = pic.new_param('A2', A2a )
  A3 = pic.new_param('A3', A3a )
  I = pic.new_param('I', sp.eye(2) )

  sdp = pic.Problem()
  Xp = sdp.add_variable('X', (2, 2), vtype='symmetric')
  sdp.add_constraint(Xp-I>>0)
  sdp.add_constraint(Xp*A1+A1.T*Xp << 0)
  sdp.add_constraint(Xp*A2+A2.T*Xp << 0)
  sdp.add_constraint(Xp*A3+A3.T*Xp << 0)

  sdp.set_objective("min", pic.trace(Xp))

  print(sdp)
  sdp.solve()
  print(Xp.value)

if __name__ == '__main__':
    #testSolveLyapnovEq()
    #testSolveLP()
    #testSolveQP()
    testSolveMinEigVal()
