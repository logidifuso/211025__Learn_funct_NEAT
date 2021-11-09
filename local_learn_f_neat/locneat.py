import configparser
import importlib
import multiprocessing
import neat
import os
import random
import signal
import sys
import time

# Local imports - Namespace packages?
# Este módulo se llama desde main.py, éste módulo no contiene ninguna informacion
# de package, por tanto se resuelve como si fuera top-level, es decir desde el
# directorio "top"
import Intercambio.local_learn_f_neat.exp1.exp1_model as sin
import Intercambio.local_learn_f_neat.common.utils as utils
import Intercambio.local_learn_f_neat.common.visualize as vis


# TODO: Tienes que modificar todas estas carpetas para crearlas (si no existen) a partir de la ruta del experimento!!!
# The current working directory
#local_dir = os.path.dirname(__file__)
# local_dir = "./"                                    # Directoria actual
# out_dir = os.path.join(local_dir, 'checkpoints') #Original, lo reemplazo por:
# The directory to store outputs
#outputs_dir = os.path.join(local_dir, 'outputs')
#graphs_dir = os.path.join(outputs_dir, 'graphs')


def eval_genomes_mp(genomes, config):
    net = neat.nn.FeedForwardNetwork.create(genomes, config)

    caso_modulo = importlib.import_module(''.join([".", ".".join(('funciones', caso))]),
                                          package='local_learn_f_neat')
    genomes.fitness = caso_modulo.eval_fitness(net)
    return genomes.fitness


def eval_genomes_single(genomes, config):
    # single process
    for genome_id, genome in genomes:
        # net = RecurrentNet.create(genome, config,1)
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        caso_modulo = importlib.import_module(''.join([".", ".".join(('funciones', caso))]),
                                              package='local_learn_f_neat')
        genome.fitness = caso_modulo.eval_fitness(net)


def create_pool_and_config(config_file, checkpoint):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    if checkpoint is not None:
        p = neat.Checkpointer.restore_checkpoint(checkpoint)
    else:
        p = neat.Population(config)

    return p, config


def evaluate_best_net(net, config):
    """
    Pequenna función para evaluar el mejor genoma encontrado durante la
    ejecución del experimento

    :param net: genema del mejor individuo encontrado
    :param config: ruta al archivo de configuración del experimento
    :return: True en caso de éxito; False en caso contrario
    """
    caso_modulo = importlib.import_module(''.join([".", ".".join(('funciones', caso))]),
                                          package='local_learn_f_neat')

    fitness = caso_modulo.eval_fitness(net)
    if fitness < config.fitness_threshold:
        return False
    else:
        return True









def run_experiment(path_results, graphs_path, checkpoints_path, config_file, caso,
                   checkpoint=None, mp=False, num_generaciones=10):

    caso_modulo = importlib.import_module(''.join([".", ".".join(('funciones', caso))]),
                                          package='local_learn_f_neat')

    p, config = create_pool_and_config(config_file, checkpoint)

    # Add a stdout reporter to show progress in the terminal.
    #p.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)


    # 2 opciones para guardar checkpointers:
    #   a) Grabar checkpoint sólo cuando el fitness del mejor individuo ha mejorado
    #   b) Grabar checkpoint cada cierto num_generaciones especificado (primer atributo de la Función
    #      Checkpointer
    # Comment out la opción descartada

    # p.add_reporter(CheckpointerBest(filename_prefix="".join((outputs_dir, '/sin_exp-checkpoint-'))))
    p.add_reporter(neat.Checkpointer(5, filename_prefix="".join((checkpoints_path, '\\checkpoint-'))))

    pe = None
    # this part is required to handle keyboard intterrupt correctly, and return population and config to
    # evaluate test set.
    try:

        if mp:
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

            pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genomes_mp)

            signal.signal(signal.SIGINT, original_sigint_handler)

            """set_trace()
            return"""
            best_genome = p.run(pe.evaluate, num_generaciones)
        else:

            best_genome = p.run(eval_genomes_single, num_generaciones)

        # Muestra info del mejor genoma
        print('\nBest genome:\n{!s}'.format(best_genome))

        # Comprobación de si el mejor genoma es un hit
        net = neat.nn.FeedForwardNetwork.create(best_genome, config)
        print("\n\nRe-evaluación del mejor individuo")
        hit = evaluate_best_net(net, config)
        if hit:
            print("ÉXITO!!!")
        else:
            print("FRACASO!!!")

        # Visualiza los resultados del experimento
        node_names = {-1: 'x', 0: 'output'}
        vis.draw_net(config, best_genome, True, node_names=node_names, directory=graphs_path, fmt='svg')
        vis.plot_stats_sine(stats, ylog=False, view=True, filename=os.path.join(graphs_path, 'avg_fitness.svg'))
        vis.plot_species(stats, view=True, filename=os.path.join(graphs_path, 'speciation.svg'))
        caso_modulo.plot_salida(net, view=True, filename=os.path.join(graphs_path, 'salida.svg'))

    except:
        print("Stopping the Jobs. ", sys.exc_info())
        if mp:
            pe.pool.terminate()
            pe.pool.join()
            print("pool ok")


        return p, config

    return p, config


#######################################################################################################
#######################################################################################################
#######################################################################################################


def experiment_configuration_parsing(ruta_config_exp):



    config_exp = configparser.ConfigParser()
    config_exp.read(ruta_config_exp)

    exper_type = config_exp.get('seccion_0', 'experimento')
    config_file_neat = config_exp.get('seccion_0', 'archivo_config_neat')
    cp = config_exp.getint('seccion_0', 'checkpoint_start')
    n_generaciones = config_exp.getint('seccion_0', 'num_generaciones')
    mp = config_exp.get('seccion_0', 'mp')
    seed = config_exp.get('seccion_0', 'seed')

    return [exper_type, config_file_neat, cp, n_generaciones, mp, seed]


def start_experiment(ruta_config_exp, ruta_experiment):

    settings = experiment_configuration_parsing(ruta_config_exp)

    # todo: podría eliminar caso como global variable si paso las funciones de evaluación al
    #  modulo correspondiente para cada caso (e.g. seno.py, funcioncita.py, etc..). Se repite
    #  código pero es más seguro...
    global caso
    caso = settings[0]
    config_file_neat = os.path.join(ruta_experiment, settings[1])
    cp = settings[2]
    n_generaciones = settings[3]
    mp = settings[4]
    seed = settings[5]
    
    path_results = os.path.join(ruta_experiment, 'resultados')
    # Limpia los resultados de la ejecución anterior (si los hubiera) o crea la carpeta a usar para guardarlos
    utils.clear_output(path_results)

    # paths a carpetas graphs y chekpoints
    graphs_path = os.path.join(path_results, 'graphs')
    checkpoints_path = os.path.join(path_results, 'checkpoints')

    # Limpia los resultados de la ejecución anterior (si los hubiera) o crea la carpeta a usar para guardarlos
    #utils.clear_output(graphs_path)
    #utils.clear_output(checkpoints_path)
    if not os.path.exists(graphs_path):
        os.makedirs(graphs_path)
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    # Fijo semilla para reproducibilidad
    random.seed(seed)

    # todo: esta forma de medir el tiempo está mal porque incluye el tiempo que tardo en mirar los gráficos.
    #  Pasarlo a la funcion run_experiment
    begin = time.time()
    if cp is not None:
        ret = run_experiment(path_results, graphs_path, checkpoints_path, config_file_neat, caso,
                             checkpoint="".join((checkpoints_path, '/checkpoint-{}'.format(cp))),
                             mp=mp, num_generaciones=n_generaciones)
    else:
        ret = run_experiment(path_results, graphs_path, checkpoints_path, config_file_neat, caso,
                             mp=mp, num_generaciones=n_generaciones)
    end = time.time()
    # Tiempo de ejecución de neat
    print(f'Tiempo de ejecución del algoritmo: {end - begin}')
    print("mp era:", mp)


    # hay que generar los corresondientes exp_model files, .. no sería más claro hacer un if para
    # seleccionar el modulo a cargar ?
    # caso_modulo = importlib.import_module(''.join([".", ".".join((caso, caso))]),
    #                                         package='local_learn_f_neat')
    #  caso_modulo.experimento(ruta_experiment, config_file_neat, cp, n_generaciones, mp, seed)
