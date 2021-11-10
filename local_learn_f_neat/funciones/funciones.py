#
# Implementación de la evaluación del fitness para función seno
#
import numpy as np
import math as m
import matplotlib.pyplot as plt

'''
def eval_seno(net):
    """
    Función que evalúa el fitness del fenotipo producido  dada una cierta ANN
    :param net: The ANN of the phenotype to be evaluated
    :return fitness: El fitness se calcula como 1/e**(-MSE). De esta forma cuando
    el error cuadrático medio es 0 el fitness es 0 y conforme aumenta el fitness
    disminuye tendiendo a cero conforme el MSE tiende a infinito
    """

    n_eval_points = 90
    err = np.ndarray([n_eval_points])

    for i in range(n_eval_points):
        valor = i*(2*np.pi/n_eval_points)
        # Normalizo el seno entre 0 y 1 para que la red pueda predecirlo (neurona de
        # salida output en [0, 1]
        err[i] = ((np.sin(valor)+1)/2 - net.activate([valor]))**2

    mse = 1/n_eval_points * np.sum(err)
    fitness = m.exp(-mse)
    return fitness


# ===================================================================================
# Crea muestras del seno escogiendo 45 muestras de ángulo entre 0 y 360 al azar para
# usar en las evaluaciones del fitness
# ------------------------------------------------------------------------------------
# create full sin list 1 step degrees
degrees2radians = np.radians(np.arange(0, 360, 1))
# samples
sample_count = 45
xx = np.random.choice(degrees2radians, sample_count, replace=False)
yy = np.sin(xx)
# ====================================================================================


def eval_seno_b(net):
    # TODO: se utilizan está variables global y las otras o no? Si no, borrar
    # global gens

    # error_sum = 0.0
    # outputs = []
    # accs = []

    def _imp():
        _fitness = 0
        for xi, xo in zip(xx, yy):
            output = net.activate([xi])
            xacc = 1 - abs(xo - output)
            _fitness += xacc

        _fitness = np.mean((_fitness / len(xx)))

        return _fitness

    fitness = (_imp()) * 100
    fitness = np.round(fitness, decimals=4)
    fitness = max(fitness, -1000.)

    return fitness


def plot_salida_seno(net, view=False, filename='salida.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    n_eval_points = 90
    salida = np.ndarray([n_eval_points])
    seno = np.ndarray([n_eval_points])
    for i in range(n_eval_points):
        valor = i * (2 * np.pi / n_eval_points)
        # val = net.activate([valor])
        salida[i] = net.activate([valor])[0]
        seno[i] = (np.sin(valor)+1)/2

    x = range(n_eval_points)

    plt.plot(x, salida, 'b-', label="salida")
    plt.plot(x, seno, 'r-', label="seno")

    plt.title("Salida vs Valor exacto")
    plt.xlabel("x")
    plt.ylabel("Salida")
    plt.grid()

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


'''


#
#Funciones.eval_fitness(net, caso)


class Funciones:

    def __init__(self, caso: str):
        """Constructor"""
        self.caso = caso

    @staticmethod
    def eval_seno(self, net):
        """
        Función que evalúa el fitness del fenotipo producido  dada una cierta ANN
        :param self:
        :param net: The ANN of the phenotype to be evaluated
        :return fitness: El fitness se calcula como 1/e**(-MSE). De esta forma cuando
        el error cuadrático medio es 0 el fitness es 0 y conforme aumenta el fitness
        disminuye tendiendo a cero conforme el MSE tiende a infinito
        """

        n_eval_points = 90
        err = np.ndarray([n_eval_points])

        for i in range(n_eval_points):
            valor = i * (2 * np.pi / n_eval_points)
            # Normalizo el seno entre 0 y 1 para que la red pueda predecirlo (neurona de
            # salida output en [0, 1]
            err[i] = ((np.sin(valor) + 1) / 2 - net.activate([valor])) ** 2

        mse = 1 / n_eval_points * np.sum(err)
        fitness = m.exp(-mse)
        return fitness

    @staticmethod
    def plot_salida_seno(self, net, view=False, filename='salida.svg'):
        """ Plots the population's average and best fitness. """
        if plt is None:
            warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
            return

        n_eval_points = 90
        salida = np.ndarray([n_eval_points])
        seno = np.ndarray([n_eval_points])
        for i in range(n_eval_points):
            valor = i * (2 * np.pi / n_eval_points)
            # val = net.activate([valor])
            salida[i] = net.activate([valor])[0]
            seno[i] = (np.sin(valor) + 1) / 2

        x = range(n_eval_points)

        plt.plot(x, salida, 'b-', label="salida")
        plt.plot(x, seno, 'r-', label="seno")

        plt.title("Salida vs Valor exacto")
        plt.xlabel("x")
        plt.ylabel("Salida")
        plt.grid()

        plt.savefig(filename)
        if view:
            plt.show()

        plt.close()

    @staticmethod
    def eval_fitness(net, caso):
        do = f"eval_{caso}"
        if hasattr(self, do) and callable(func := getattr(self, do)):
            func(net)

    @staticmethod
    def plot_salida(net, caso):
        do = f"plot_salida_{caso}"
        if hasattr(self, do) and callable(func := getattr(self, do)):
            func(net)


"""
    def eval_fitness(self, net, caso):
        do = f"eval_{caso}"
        if hasattr(self, do) and callable(func := getattr(self, do)):
            func(net)
"""

class Prueba:

    def __init__(self, valor: int):
        """Constructor"""
        self.valor = valor


    def valor0(self):
        print("Escogiste valor 0")

    def valor1(self):
        print("Escogiste valor 1")

    def escoge(self):
        if self.valor == 0:
            self.valor0()
        elif self.valor == 1:
            self.valor1()
        else:
            print("Ni lo uno ni lo otro")

    def devuelve_metodo(self):
        x = self.valor
        do = f"valor{x}"
        if hasattr(self, do) and callable(getattr(self, do)):
            func = getattr(self, do)
            return func





