### Решение двумерного уравнения Гельмгольца в квадратной области с нулевыми граничными условиями и заданной правой частью

Для численного решения использовалась конечно-разностная схема "крест" второго порядка апроксимации по обеим независимым переменным.
Полученная СЛАУ решается иетарционным методом Якоби и его модификацией ("красно-черные" итерации).

При использовании MPI были задействованы три способа пересылок:

Send-Recive

SendRecive

Isend-Irecive
