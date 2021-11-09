import configparser
import importlib
import os

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


def llamada_a_experimento(ruta_config_exp, ruta_experiment):

    settings = experiment_configuration_parsing(ruta_config_exp)

    caso = settings[0]
    config_file_neat = os.path.join(ruta_experiment, settings[1])
    cp = settings[2]
    n_generaciones = settings[3]
    mp = settings[4]
    seed = settings[5]

    # TODO: Ya que puedo probar diferentes funciones pero siempre sabré cuales son de antemano porque
    # hay que generar los corresondientes exp_model files, .. no sería más claro hacer un if para
    # seleccionar el modulo a cargar ?
    caso_modulo = importlib.import_module(''.join([".", ".".join((caso, caso))]),
                                           package='local_learn_f_neat')
    caso_modulo.experimento(ruta_experiment, config_file_neat, cp, n_generaciones, mp, seed)
