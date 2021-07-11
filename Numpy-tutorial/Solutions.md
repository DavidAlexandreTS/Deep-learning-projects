# NumPy Array
- How to create an empty and a full NumPy array?<br>
***numpy.empty(shape, dtype = float, order = ‘C’) e numpy.full(shape, fill_value, dtype = None, order = ‘C’) respectivamente***
- Create a Numpy array filled with all zeros<br>
***a = np.zeros(3, dtype = int) considerando que temos um import numpy as np(para esta e para as demais questões)***
- Create a Numpy array filled with all ones<br>
***De forma análoga a anterior temos a = np.ones(3, dtype = int) ***
- Check whether a Numpy array contains a specified row<br>
***Considerando arr como um vetor numpy, basta executarmos print([1, 2, 3, 4, 5] in arr.tolist())***
- How to Remove rows in Numpy array that contains non-numeric values?<br>
***print(arr[~np.isnan(arr).any(axis=1)])***
- Remove single-dimensional entries from the shape of an array<br>
***Considere a declaração x = np.zeros((3, 1, 4)), basta então fazermos print(np.squeeze(x).shape)***
- Find the number of occurrences of a sequence in a NumPy array<br>
***output = repr(arr).count("9, 4")***
- Find the most frequent value in a NumPy array<br>
***print(np.bincount(x).argmax())***
- Combining a one and a two-dimensional NumPy Array<br>
***Para combinar precisamos fazer um for a, b in np.nditer([num_1d, num_2d]): e em seguida um print("%d:%d" % (a, b),)***
- How to build an array of all combinations of two NumPy arrays?<br>
***comb_array = np.array(np.meshgrid(array_1, array_2)).T.reshape(-1, 2)***
- How to add a border around a NumPy array?<br>
***numpy.pad(array, pad_width, mode='constant', **kwargs)***
- How to compare two NumPy arrays?<br>
***Para realizarmos esta tarefa fazemos comparison = an_array == another_array e em seguida equal_arrays = comparison.all() o que nos leva a finalizar com print(equal_arrays)***
- How to check whether specified values are present in NumPy array?<br>
***Fazemos simples mente um if value in my_array[:, col_num]: que se for verdadeiro acha os valores***
- How to get all 2D diagonals of a 3D NumPy array?<br>
***numpy.diagonal(a, axis1, axis2)***
- Flatten a Matrix in Python using NumPy<br>
***Depois de declaradas as matrizes precisamos apenas fazer flat_gfg = gfg.flatten()***
- Flatten a 2d numpy array into 1d array<br>
***Podemos fazer usando o np.flatten, assim result = ini_array1.flatten()***
- Move axes of an array to new positions<br>
***Podemos fazer simplesmente gfg = np.moveaxis(arr, 0, -1).shape***
- Interchange two axes of an array<br>
***De forma análoga gfg = np.swapaxes(arr, 0, 1)***
- NumPy – Fibonacci Series using Binet Formula<br>
***import numpy as np
  We are creating an array contains n = 10 elements
  for getting first 10 Fibonacci numbers
  a = np.arange(1, 11)
  lengthA = len(a)
  splitting of terms for easiness
  sqrtFive = np.sqrt(5)
  alpha = (1 + sqrtFive) / 2
  beta = (1 - sqrtFive) / 2
  Implementation of formula
  np.rint is used for rounding off to integer
  Fn = np.rint(((alpha ** a) - (beta ** a)) / (sqrtFive))
  print("The first {} numbers of Fibonacci series are {} . ".format(lengthA, Fn))***
- Counts the number of non-zero values in the array<br>
***gfg = np.count_nonzero(arr)***
- Count the number of elements along a given axis<br>
***print(np.size(arr))***
- Trim the leading and/or trailing zeros from a 1-D array<br>
***numpy.trim_zeros(arr, trim)***
- Change data type of given numpy array<br>
***arr = arr.astype('float64')***
- Reverse a numpy array<br>
***Um dos métodos que podem ser usados seria res = ini_array[::-1]***
- How to make a NumPy array read-only?<br>
***array.flags.writeable=False***
# Questions on NumPy Matrix
- Get the maximum value from given matrix<br>
***rslt1 = np.amax(arr, 1)***
- Get the minimum value from given matrix<br>
***rslt2 = np.amin(arr, 1)***
- Find the number of rows and columns of a given matrix using NumPy<br>
***print(matrix.shape)***
- Select the elements from a given matrix<br>
***matrix.choose()***
- Find the sum of values in a matrix<br>
***numpy.ndarray.sum()***
- Calculate the sum of the diagonal elements of a NumPy array<br>
***trace = np.trace(n_array)***
- Adding and Subtracting Matrices in Python<br>
***print(np.add(A, B)) e print(np.subtract(A, B))***
- Ways to add row/columns in numpy array<br>
***result = np.hstack((ini_array, np.atleast_2d(column_to_be_added).T))***
- Matrix Multiplication in NumPy<br>
***res = np.dot(mat1,mat2)***
- Get the eigen values of a matrix<br>
***A sintaxe para fazermos isto é numpy.linalg.eig()***
- How to Calculate the determinant of a matrix using NumPy?<br>
***numpy.linalg.det(array)***
- How to inverse a matrix using NumPy<br>
***Basicamente fazemos um if det(A) != 0 e caso isso seja verdade a matriz pode ser invertida usando numpy.linalg.inv(a)***
- How to count the frequency of unique values in NumPy array?<br>
***numpy.unique(arr, return_counts=False)***
- Multiply matrices of complex numbers using NumPy in Python<br>
***Usando numpy.vdot(vector_a, vector_b)***
- Compute the outer product of two given vectors using NumPy in Python<br>
***Considerando que os parametros são válidos result = np.outer(array1, array2)***
- Calculate inner, outer, and cross products of matrices and vectors using NumPy<br>
***numpy.inner(arr1, arr2)***
- Compute the covariance matrix of two given NumPy arrays<br>
***np.cov(array1, array2)***
- Convert covariance matrix to correlation matrix using Python<br>
***Podemos implementar uma função def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation***
- Compute the Kronecker product of two mulitdimension NumPy arrays<br>
***numpy.kron(array1, array2)***
- Convert the matrix into a list<br>
***Podemos usar a tolist(), para isso fazemos y = x.tolist()***
# Questions on NumPy Indexing
- Replace NumPy array elements that doesn’t satisfy the given condition<br>
***De varias formas, uma delas seria n_arr[n_arr > 50.] = 15.50 para substituir os elementos maiores que 50***
- Return the indices of elements where the given condition is satisfied<br>
***numpy.where(condition[, x, y])***
- Replace NaN values with average of columns<br>
***a[inds] = np.take(col_mean, inds[1])***
- Replace negative value with zero in numpy array<br>
***ini_array1[ini_array1<0] = 0***
- How to get values of an NumPy array at certain index positions?<br>
***print("\nTake 1 and 15 from Array 2 and put them in\
1st and 5th position of Array 1") e em seguida a1.put([0, 4], a2)***
- Find indices of elements equal to zero in a NumPy array<br>
***numpy.where(condition[, x, y])***
- How to Remove columns in Numpy array that contains non-numeric values?<br>
***Fazemos apenas print(n_arr[:, ~np.isnan(n_arr).any(axis=0)])***
- How to access different rows of a multidimensional NumPy array?<br>
***res_arr = arr[[0,2]] pode ser uma solução, mas vai depender da dimensão(neste caso é 2)***
- Get row numbers of NumPy array having element larger than X<br>
***numpy.where(numpy.any(arr > X,
                                axis = 1))***
- Get filled the diagonals of NumPy array<br>
***numpy.fill_diagonal(array, value)***
- Check elements present in the NumPy array<br>
***Podem existir outras formas, mas uma intuitiva e simples se resume a print(2 in n_array) e aí a saída True nos diz que ele está no array***
- Combined array index by index<br>
***Podemos simplesmente concatena-los fazendo C = A + B***
# Questions on NumPy Linear Algebra
- Find a matrix or vector norm using NumPy<br>
***vec_norm = np.linalg.norm(vec)***
- Calculate the QR decomposition of a given matrix using NumPy<br>
***q, r = np.linalg.qr(matrix1)***
- Compute the condition number of a given matrix using NumPy<br>
***np.linalg.cond(matrix)***
- Compute the eigenvalues and right eigenvectors of a given square array using NumPy?<br>
***A sintaxe aqui é simples, sendo apenas numpy.linalg.eig()***
- Calculate the Euclidean distance using NumPy<br>
***dist = np.linalg.norm(point1 - point2)***
# Questions on NumPy Random
- Create a Numpy array with random values<br>
***b = geek.empty(2, dtype = int)***
- How to choose elements from the list with different probability using NumPy?<br>
***number = np.random.choice(num_list)***
- How to get weighted random choice in Python?<br>
***randomList = random.choices(sampleList, weights=(10, 20, 30, 40, 50), k=5)***
- Generate Random Numbers From The Uniform Distribution using NumPy<br>
***r = np.random.uniform(size=4)***
- Get Random Elements form geometric distribution<br>
***gfg = np.random.geometric(0.65, 1000)
  count, bins, ignored = plt.hist(gfg, 40, density = True)
  plt.show()***
- Get Random elements from Laplace distribution<br>
***gfg = np.random.laplace(1.45, 15, 1000)
  count, bins, ignored = plt.hist(gfg, 30, density = True)
  plt.show()***
- Return a Matrix of random values from a uniform distribution<br>
***gfg = np.random.uniform(-5, 5, 5000)
  plt.hist(gfg, bins = 50, density = True)
  plt.show()***
- Return a Matrix of random values from a Gaussian distribution<br>
***Através da sintaxe simples random.normal(loc=0.0, scale=1.0, size=None)***
# Questions on NumPy Sorting and Searching
- How to get the indices of the sorted array using NumPy in Python?<br>
***A sintaxe para o comando é basicamente numpy.argsort(arr, axis=-1, kind=’quicksort’, order=None)***
- Finding the k smallest values of a NumPy array<br>
***Primeiro adicionamos um valor a k, k = 4 depois ordenamos arr1 = np.sort(arr) e por fim fazemos print(k, "smallest elements of the array") seguido na outra linha de print(arr1[:k])***
- How to get the n-largest values of an array using NumPy?<br>
***Podemos seguir aqui de forma analoga a que fizemos anteriormente so que agora com n, dai rslt = sorted_array[-n : ] e por fim print("{} largest value:".format(n), rslt[0])***
- Sort the values in a matrix<br>
***matrix.sort()***
- Filter out integers from float numpy array<br>
***result = ini_array[ini_array != ini_array.astype(int)]***
- Find the indices into a sorted array<br>
***indices = np.argsort(array)***
# Questions on NumPy Mathematics
- How to get element-wise true division of an array using Numpy?<br>
***rslt = np.true_divide(x, 4)***
- How to calculate the element-wise absolute value of NumPy array?<br>
***numpy.absolute(arr, out = None, ufunc ‘absolute’)***
- Compute the negative of the NumPy array<br>
***out_num = np.negative(in_num)  seguido de um print ("negative of input number : ", out_num) ***
- Multiply 2d numpy array corresponding to 1d array<br>
***result = ini_array1 * ini_array2[:, np.newaxis]***
- Computes the inner product of two arrays<br>
***numpy.dot(vector_a, vector_b, out = None)***
- Compute the nth percentile of the NumPy array<br>
***numpy.percentile(arr, n, axis=None, out=None) ***
- Calculate the n-th order discrete difference along the given axis<br>
***numpy.diff()***
- Calculate the sum of all columns in a 2D NumPy array<br>
***Podemos implementar uma função para isto 
def colsum(arr, n, m):
    for i in range(n):
        su = 0;
        for j in range(m):
            su += arr[j][i]
        print(su, end = " ")***
- Calculate average values of two given NumPy arrays<br>
***avg = (arr1 + arr2) / 2***
- How to compute numerical negative value for all elements in a given NumPy array?<br>
***r1 = np.negative(x)***
- How to get the floor, ceiling and truncated values of the elements of a numpy array?<br>
***A sintaxe por trás disto é numpy.floor(x[, out]) = ufunc ‘floor’) ***
- How to round elements of the NumPy array to the nearest integer?<br>
***y = n.rint(y)***
- Find the round off the values of the given matrix<br>
***Podemos usar o matrix.round()***
- Determine the positive square-root of an array<br>
***Usando o numpy.sqrt()***
- Evaluate Einstein’s summation convention of two multidimensional NumPy arrays<br>
***numpy.einsum(subscripts, *operands, out=None)***
# Questions on NumPy Statistics
- Compute the median of the flattened NumPy array<br>
***numpy.median(arr, axis = None)***
- Find Mean of a List of Numpy Array<br>
***Podemos usar o np.mean, daí 
  for i in range(len(Input)):
   Output.append(np.mean(Input[i]))***
- Calculate the mean of array ignoring the NaN value<br>
***arr = np.array([[20, 15, 37], [47, 13, np.nan]])
   print("Shape of array is", arr.shape)
   print("Mean of array without using nanmean function:", np.mean(arr))
   print("Using nanmean function:", np.nanmean(arr))***
- Get the mean value from given matrix<br>
***matrix.mean(axis=None, dtype=None, out=None)***
- Compute the variance of the NumPy array<br>
***Existem algumas formas, uma delas seria rint("var of arr : ", np.var(arr)) ***
- Compute the standard deviation of the NumPy array<br>
***# 1D array 
arr = [20, 2, 7, 1, 34]
  print("arr : ", arr) 
  print("std of arr : ", np.std(arr))
  print ("\nMore precision with float32")
  print("std of arr : ", np.std(arr, dtype = np.float32))
  print ("\nMore accuracy with float64")
  print("std of arr : ", np.std(arr, dtype = np.float64))***
- Compute pearson product-moment correlation coefficients of two given NumPy arrays<br>
***rslt = np.corrcoef(array1, array2)***
- Calculate the mean across dimension in a 2D NumPy array<br>
***Uma solução simplista seria 
# Calculating mean across Rows
row_mean = np.mean(arr, axis=1)
row1_mean = row_mean[0]
print("Mean of Row 1 is", row1_mean) no entanto aqui calculamos apenas 1 linha, o processo para as demasi é analogo***
- Calculate the average, variance and standard deviation in Python using NumPy<br>
***Aqui usamos print(np.average(list))***
- Describe a NumPy Array in Python<br>
***Um vetor numpy é uma grade de valores, todos do mesmo tipo, e é indexada por inteiros não negativos***
# Questions on Polynomial
- Define a polynomial function<br>
***Uma função polinomial é aquela que é definida por uma expressão polinomial***
- How to add one polynomial to another using NumPy in Python?<br>
***px = (5,-2,5)
  qx = (2,-5,2)
  rx = numpy.polynomial.polynomial.polyadd(px,qx)***
- How to subtract one polynomial to another using NumPy in Python?<br>
***De forma analoga ao que fizemos acima basta seguir e substituir a linha 3 por rx = numpy.polynomial.polynomial.polysub(px,qx)***
- How to multiply a polynomial to another using NumPy in Python?<br>
***Novamente ... rx = numpy.polynomial.polynomial.polymul(px, qx)***
- How to divide a polynomial to another using NumPy in Python?<br>
***qx, rx = numpy.polynomial.polynomial.polydiv(px, gx)***
- Find the roots of the polynomials using NumPy<br>
***numpy.roots(p)***
- Evaluate a 2-D polynomial series on the Cartesian product<br>
***np.polygrid2d(x, y, c)***
- Evaluate a 3-D polynomial series on the Cartesian product<br>
***np.polygrid3d(x, y, z, c)***
# Questions on NumPy Strings
- Repeat all the elements of a NumPy array of strings<br>
***new_array = np.char.multiply(arr, 3)***
- How to split the element of a given NumPy array with spaces?<br>
***sparr = np.char.split(array)***
- How to insert a space between characters of all the elements of a given NumPy array?<br>
***r = np.char.join(" ", x)***
- Find the length of each string element in the Numpy array<br>
***Primeiro fazemos length_checker = np.vectorize(len) depois arr_len = length_checker(arr) e por fim print(arr_len)***
- Swap the case of an array of string<br>
***numpy.char.swapcase(arr)***
- Change the case to uppercase of elements of an array<br>
***out_arr = np.char.upper(in_arr)***
- Change the case to lowercase of elements of an array<br>
***out_arr = np.char.lower(in_arr)***
- Join String by a seperator<br>
***out_arr = geek.core.defchararray.join(sep, in_arr)***
- Check if two same shaped string arrayss one by one<br>
***out_arr = geek.char.equal(in_arr1, in_arr2)***
- Count the number of substrings in an array<br>
***Podemos implementar uma função para tal def countNonEmptySubstr(str):
    n = len(str);
    return int(n * (n + 1) / 2);***
- Find the lowest index of the substring in an array<br>
***char.find(a, sub, start=0, end=None)***
- Get the boolean array when values end with a particular character<br>
***a = np.array(['geeks', 'for', 'geeks'])
  gfg = np.char.endswith(a, 'ks')***
- More Questions on NumPy<br>
***Não entendi essa professor, perdão***
- Different ways to convert a Python dictionary to a NumPy array<br>
***numpy.array(object, dtype = None, *, copy = True, order = ‘K’, subok = False, ndmin = 0)***
- How to convert a list and tuple into NumPy arrays?<br>
***numpy.asarray(  a, type = None, order = None )***
- Ways to convert array of strings to array of floats<br>
***x = np.array(['1.1', '2.2', '3.3'])
y = x.astype(np.float)***
- Convert a NumPy array into a csv file<br>
***Usando o pandas como pd, temos inicialmente DF = pd.DataFrame(arr) e em seguida DF.to_csv("data1.csv") para salvar o dataframe***
- How to Convert an image to NumPy array and save it to CSV file using Python?<br>
***Podemos fazer usando o PIl e em seguida imageToMatrice = gfg.asarray(img)***
- How to save a NumPy array to a text file?<br>
***Inicialmente precisamos abrir o arquivo com file = open("file1.txt", "w+") e depois salvar o array n oarquivo texto com content = str(Array)
  file.write(content)
  file.close()***
- Load data from a text file<br>
***Para carregar os dados é só fazer file = open("file1.txt", "r")
  content = file.read()***
- Plot line graph from NumPy array<br>
***plt.title("Line graph")
  plt.xlabel("X axis")
  plt.ylabel("Y axis")
  plt.plot(x, y, color ="red")
  plt.show()***
- Create Histogram using NumPy<br>
***numpy.histogram(data, bins=10, range=None, normed=None, weights=None, density=None)***
