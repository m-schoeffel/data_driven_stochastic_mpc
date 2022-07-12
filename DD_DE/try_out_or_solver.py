from ast import Expression
from ortools.sat.python import cp_model

model = cp_model.CpModel()

x = model.NewIntVar(-100, 100, 'x')
y = model.NewIntVar(-100, 100, 'y')

x_2 = model.NewIntVar(0, 100000, 'x')
y_2 = model.NewIntVar(0, 100000, 'y')

my_var = list()
for i in range(0,5):
    my_var.append(model.NewIntVar(0, 100000, "My_Var "+str(i)))

def add_const(list_var):
    my_const=0
    for var in list_var:
        my_const = my_const+var
    my_const = my_const >= 4
    
    print(my_const)
    return my_const

if(add_const([5,2])):
    print("It works! :-)")
else:
    print("too small :<(")

model.Add(x+y <= 8)
model.Add(y <= 5)
model.Add(add_const([x,y]))
model.Add(add_const([my_var[0],my_var[1]]))

model.AddMultiplicationEquality(x_2, [x, x])
model.AddMultiplicationEquality(y_2, [y, y])

def my_expression(a,b):
    return a+b

model.Maximize(my_expression(my_var[0],my_var[1]))


solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f'Maximum of objective function: {solver.ObjectiveValue()}\n')
    print(f'x = {solver.Value(x)}')
    print(f'y = {solver.Value(y)}')
else:
    print('No solution found.')
