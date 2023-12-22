# Optimization_Algorithms
My codes for solving Group Lasso Problem using various optimization algorithms

本次Project主要围绕下述 Group Lasso 优化问题进行展开, 借助Matlab编译器， 分别使用了如下方法该问题进行优化.

首先，为设置目标与对比， 尝试调用专业的优化求解器mosek与gurobi进行求解。

    1. 调用Matlab中CVX工具包的mosek求解器进行求解.
    2. 调用Matlab中CVX工具包的gurobi求解器进行求解.
    3. 直接调用mosek求解器进行求解.
    4. 直接调用gurobi求解器进行求解.


之后， 尝试根据现有算法自行设置求解器与求解过程，使用了如下算法.


    1. 次梯度下降算法求解原问题.
    2. 邻近算子梯度法求解原问题.
    3. 快速邻近算子梯度法求解原问题.
    4. 增广拉格朗日函数法求解对偶问题.
    5. 交替方向乘子法求解对偶问题.
    6. 增广拉格朗日函数法求解线性化原问题.

实验报告与代码将附在文件中，欢迎批评指正！

感谢阅读！
