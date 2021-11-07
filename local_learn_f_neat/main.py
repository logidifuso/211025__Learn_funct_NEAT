
import argparse
import configparser
import importlib

if __name__ == '__main__':

    # ----------------------------------------------------------------------------
    # Parsing de los argumentos, ó lectura de la configuración del experimento y
    # del algoritmo Neat desde los archivos de configuración ".ini"
    # ----------------------------------------------------------------------------
    ap = argparse.ArgumentParser()

    ap.add_argument("-exp", "--experimento", required=False,
                    help="Indica el experimento a realizar")

    ap.add_argument("-cfg_neat", "--config_file_neat", required=False,
                    help="nombre del archvo de configuración de Neat")

    ap.add_argument("-cfg_exp", "--config_file_exp", required=False,
                    help="nombre del archvo de configuración del experimento")

    ap.add_argument("-cp", "--checkpoint_start", required=False, type=int,
                    help="número del checkpoint a usar")

    ap.add_argument("-gen", "--num_generaciones", required=False, type=int,
                    help="número de generaciones a correr")

    args = vars(ap.parse_args())

    # Fija el experimento a realizar. Si no se indica nada en parsing, se asume
    # siempre exp1
    if args['experimento'] is not None:
        caso = args['experimento']
    else:
        caso = 'exp1'

    # Set nombre del archivo de configuración del experimento (si se proporciona)
    config_exp = None
    if args['config_file_exp'] is not None:
        ruta = args['config_file_exp']
        config_exp = configparser.ConfigParser()
        config_exp.read(ruta)

    # Indica el nombre del archivo de configuración para NEAT. Si no se especifica,
    # el nombre por defecto es siempre: 'config.ini'
    if args['config_file_neat'] is not None:
        config_file_neat = args['config_file_neat']
    elif config_exp is not None:
        # Se asume que cp existe en el archivo .ini, en caso contrario dará error
        config_file_neat = config_exp.get('seccion_0', 'archivo_config_neat')
    else:
        config_file_neat = 'config.ini'

    # Set cp (checpoint_start) si retoma a partir de una ejecución previa.
    if args['checkpoint_start'] is not None:
        cp = args['checkpoint_start']
    elif config_exp is not None:
        # Requiere que cp exista en el archivo .ini, en caso contrario error
        cp = config_exp.getint('seccion_0', 'checkpoint_start')
        if cp == 0:
            cp = None
    else:
        cp = None

    if args['num_generaciones'] is not None:
        n_generaciones = args['num_generaciones']
    elif config_exp is not None:
        # Requiere que cp exista en el archivo .ini, en caso contrario error
        n_generaciones = config_exp.getint('seccion_0', 'num_generaciones')
    else:
        n_generaciones = 15

    if config_exp is not None:
        # Requiere que cp exista en el archivo .ini, en caso contrario error
        mp = config_exp.get('seccion_0', 'mp')
    else:
        mp = True  # Por defecto se usa el módulo de multiprocessing

    if config_exp is not None:
        # Requiere que cp exista en el archivo .ini, en caso contrario error
        seed = config_exp.get('seccion_0', 'seed')
    else:
        seed = 1559231615  # Valor de la semilla por defecto

    # Ejecución del experimento
    caso_modulo = importlib.import_module(".".join((caso, caso)))
    caso_modulo.experimento(config_file_neat, cp, n_generaciones, mp, seed)
