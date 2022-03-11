import numpy as np


class pca():

    def __init__(self, n):
        self.n = n

    def fit(self, data):



        #Calculo de la matrix de convarinza
        cov_mat = np.cov(data.T)
        '''print('NumPy covariance matrix: \n%s' %cov_mat)'''

        #Calculo de los eigenvector y eigenvalues
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        '''
        print('Eigenvectors \n%s' %eigen_vecs)
        print('Eigenvalues \n%s' %eigen_vals)
        '''
        #Listar y orden las parejas de Eigenvectors y Eigenvalues
        eigen_pairs =  [ (np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]

        eigen_pairs.sort(key = lambda x: x[0], reverse = True)
        '''print('Eigenvalues en orden descendente')
        for i in eigen_pairs:
            print(i[0])
        '''
        #Con loos eigenvalues ordenados, tenemos cuales son los que mas relevancia tienen en la matrix original
        #Lo que sigue es escoger la cantidad de eigen values de mayor a menor que representara nuestro nuevo set 
        # (es importante que la cantidad sea representativa, por lo cual se debe poner un valor porcentual por el cual se necesita o indicar cuanta informacion se saca con la cantidad escogida)

        #continuamos generando la matrix de proyeccion a partir de los eigenvalues escogidos
        matrix_proyeccion = np.hstack((eigen_pairs[0][1].reshape(len(data.columns),1),eigen_pairs[1][1].reshape(len(data.columns),1)))
        '''print('Matriz de Proyeccion:\n',matrix_proyeccion)'''
        
        #por ultimo sacamos los nuevos componentes de los datos
        Y = data.dot(matrix_proyeccion)
        #mostramos
        '''print('PCA resutltado \n', Y.to_numpy())'''

        Y = Y.to_numpy()

        return Y